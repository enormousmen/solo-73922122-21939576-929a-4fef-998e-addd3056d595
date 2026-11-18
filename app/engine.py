"""Single-process micro-batch inference engine.

Concurrency model
-----------------
* Everything except the blocking model forward pass runs on the FastAPI
  asyncio event loop, so no locks are needed for the shared state below.
* Exactly one worker task forms batches and exactly one executor thread runs
  the TorchScript model, therefore at most ONE batch is ever actually
  inferring at any moment.
* Capacity accounting: an order occupies a slot while it is queued AND while
  its batch is computing (even if it was cancelled or expired mid-flight).
  The slot is released only when the batch compute actually finishes.
* Drain ("stop taking traffic") is a sticky flag set synchronously on the
  event loop. `submit` re-checks it in the same synchronous section that
  registers the order, so an order can never be accepted (201) after the
  drain flag is visible. Every accepted order occupies a slot (queued or
  in-flight) and is therefore counted as unsettled until it is released.
  A queued order that reaches its deadline is expired by a dedicated timer
  task even if nobody ever queries it.

State machine per order
-----------------------
queued -> processing -> completed
queued -> cancelled            (cancelled before entering a batch; never runs)
queued -> expired              (deadline passed while queued; never runs)
processing -> cancelled        (marked immediately, result discarded, slot held)
processing -> expired          (deadline passed mid-flight; result discarded)
processing -> failed           (model raised; defensive only)
Late results are written back ONLY to orders still in `processing` whose
deadline has not passed, so a late result can never revive a cancelled or
expired order.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import torch

log = logging.getLogger("batch-engine")

QUEUED = "queued"
PROCESSING = "processing"
COMPLETED = "completed"
CANCELLED = "cancelled"
EXPIRED = "expired"
FAILED = "failed"


class DrainingError(RuntimeError):
    """Raised by `submit` once draining has started (new work is rejected)."""


# eq=False: orders are compared by identity. The default dataclass __eq__
# would compare tensors element-wise and raise on multi-element tensors.
@dataclass(eq=False)
class Order:
    id: str
    tensor: torch.Tensor
    deadline_ms: int
    deadline_at: float  # time.monotonic() at which the order expires
    state: str = QUEUED
    result: Optional[dict] = None
    batch_id: Optional[str] = None
    created_at: float = field(default_factory=time.monotonic)


class BatchEngine:
    def __init__(
        self,
        model_path: str,
        max_capacity: int = 16,
        batch_size: int = 4,
        batch_wait_ms: int = 50,
        inference_delay_ms: int = 0,
    ) -> None:
        self.max_capacity = max_capacity
        self.batch_size = batch_size
        self.batch_wait_s = batch_wait_ms / 1000.0
        self.inference_delay_s = inference_delay_ms / 1000.0

        # Single-threaded CPU inference keeps results deterministic and makes
        # the "one batch at a time" guarantee easy to reason about.
        torch.set_num_threads(1)
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()

        self._pending: deque[Order] = deque()
        # Frames popped from `_pending` into the batch-formation window but not
        # yet handed to the model. They still occupy capacity (and are still
        # unsettled for drain purposes) until the batch starts or is dropped.
        self._handoff: list[Order] = []
        self._orders: dict[str, Order] = {}
        self._inflight = 0  # orders inside the batch currently being computed
        self._not_empty = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self._worker: Optional[asyncio.Task] = None
        # Wakes the expiry sweeper when new queued work (or an earlier deadline)
        # appears; keeps the idle sweeper from busy-polling.
        self._sweep_wakeup = asyncio.Event()
        # Drain state. `_draining` is only touched on the event loop, so the
        # check-and-accept in submit() is atomic with respect to POST /drain.
        self._draining = False
        self._drained_at: Optional[float] = None
        self._expiry_task: Optional[asyncio.Task] = None
        # Counters for /stats and tests.
        self.batches_started = 0
        self.last_batch_size = 0
        self.accepted_orders = 0       # every order that got a 201
        self.backlog_at_drain = 0  # unsettled frames when drain started

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        self._worker = asyncio.create_task(self._run(), name="batch-worker")
        self._expiry_task = asyncio.create_task(self._expire_loop(), name="expiry-sweeper")

    async def stop(self) -> None:
        for task in (self._worker, self._expiry_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------ #
    # helpers (event-loop thread only)
    # ------------------------------------------------------------------ #
    def _purge_expired_pending(self) -> None:
        """Mark queued orders whose deadline passed; they never enter the model."""
        now = time.monotonic()
        kept: list[Order] = []
        for o in self._pending:
            if o.state == QUEUED and o.deadline_at <= now:
                o.state = EXPIRED
            else:
                kept.append(o)
        if len(kept) != len(self._pending):
            self._pending.clear()
            self._pending.extend(kept)
        # Framed but not yet batched: expiry applies here too. They are dropped
        # from the handoff and therefore never count against capacity again.
        handoff_kept: list[Order] = []
        for o in self._handoff:
            if o.state == QUEUED and o.deadline_at <= now:
                o.state = EXPIRED
            else:
                handoff_kept.append(o)
        self._handoff = handoff_kept

    def _refresh_state(self, o: Order) -> None:
        """Lazy expiry when an order is observed."""
        if o.state in (QUEUED, PROCESSING) and o.deadline_at <= time.monotonic():
            if o.state == QUEUED:
                try:
                    self._pending.remove(o)
                except ValueError:
                    pass
                try:
                    self._handoff.remove(o)
                except ValueError:
                    pass
            o.state = EXPIRED

    def _unsettled(self) -> int:
        return len(self._pending) + len(self._handoff) + self._inflight

    def capacity_used(self) -> int:
        self._purge_expired_pending()
        return self._unsettled()

    def stats(self) -> dict:
        self._purge_expired_pending()
        counts: dict[str, int] = {}
        for o in self._orders.values():
            self._refresh_state(o)
            counts[o.state] = counts.get(o.state, 0) + 1
        return {
            "capacity_used": self._unsettled(),
            "max_capacity": self.max_capacity,
            "queued": len(self._pending) + len(self._handoff),
            "inflight": self._inflight,
            "batches_started": self.batches_started,
            "last_batch_size": self.last_batch_size,
            "orders": counts,
            "accepted_orders": self.accepted_orders,
            "draining": self._draining,
        }

    # ------------------------------------------------------------------ #
    # drain ("stop taking traffic, let accepted orders finish")
    # ------------------------------------------------------------------ #
    @property
    def draining(self) -> bool:
        return self._draining

    def drain(self) -> dict:
        """Start (or re-read) draining. Idempotent: repeated calls never reset
        the start time or the accepted-order bookkeeping."""
        if not self._draining:
            self._draining = True
            self._drained_at = time.monotonic()
            self.backlog_at_drain = self.capacity_used()
        return self.drain_status()

    def drain_status(self) -> dict:
        # Purges expired queued orders: they are settled without a query and
        # without entering the model.
        self._purge_expired_pending()
        queued = len(self._pending) + len(self._handoff)
        inflight = self._inflight
        computing = inflight > 0
        unsettled = self._unsettled()
        ready = self._draining and unsettled == 0 and not computing
        return {
            "draining": self._draining,
            "ready_to_shutdown": ready,
            "unsettled_frames": unsettled,
            "settled_frames": max(0, self.accepted_orders - unsettled),
            "queued_frames": queued,
            "computing_frames": inflight,
            "model_computing": computing,
            "idle": unsettled == 0 and not computing,
            "capacity_used": unsettled,
            "max_capacity": self.max_capacity,
            "accepted_orders": self.accepted_orders,
            "backlog_at_drain": self.backlog_at_drain,
            "drain_elapsed_ms": (
                max(0, round((time.monotonic() - self._drained_at) * 1000))
                if self._drained_at is not None
                else None
            ),
        }

    # ------------------------------------------------------------------ #
    # public API used by the HTTP layer
    # ------------------------------------------------------------------ #
    def submit(self, tensor: torch.Tensor, deadline_ms: int) -> Optional[Order]:
        """Accept an image, or return None when at capacity (16 by default).

        Raises DrainingError once draining has started. The flag check, the
        capacity check and the registration below are one synchronous section
        on the event loop, so no submit can be accepted after POST /drain has
        flipped the flag, and every order that got a 201 is already occupying
        a slot when drain inspects the backlog.
        """
        if self._draining:
            raise DrainingError("engine is draining; new submissions are rejected")
        if self.capacity_used() >= self.max_capacity:
            return None
        order = Order(
            id=uuid.uuid4().hex[:12],
            tensor=tensor,
            deadline_ms=deadline_ms,
            deadline_at=time.monotonic() + deadline_ms / 1000.0,
        )
        self._orders[order.id] = order
        self._pending.append(order)
        self.accepted_orders += 1
        self._not_empty.set()
        self._sweep_wakeup.set()
        return order

    def get(self, order_id: str) -> Optional[Order]:
        o = self._orders.get(order_id)
        if o is not None:
            self._refresh_state(o)
        return o

    def cancel(self, order_id: str) -> Optional[Order]:
        o = self._orders.get(order_id)
        if o is None:
            return None
        self._refresh_state(o)
        if o.state == QUEUED:
            try:
                self._pending.remove(o)
            except ValueError:
                pass
            try:
                self._handoff.remove(o)
            except ValueError:
                pass
            o.state = CANCELLED
        elif o.state == PROCESSING:
            # Mark immediately. The slot stays occupied until the batch
            # compute actually ends; no new batch is started because of it.
            o.state = CANCELLED
        return o

    # ------------------------------------------------------------------ #
    # batching worker
    # ------------------------------------------------------------------ #
    async def _expire_loop(self) -> None:
        """Settle queued frames at their deadline even if nobody queries them.

        Expiry also happens lazily on read and in the batching loop, but drain
        must observe an empty backlog without a query, so the sweep is timer
        driven while there is outstanding queued work. It never touches
        `_inflight`: those frames stay unsettled until the compute ends.
        """
        while True:
            self._purge_expired_pending()
            outstanding = list(self._pending) + list(self._handoff)
            if outstanding:
                nearest = min(o.deadline_at for o in outstanding)
                delay = min(0.05, max(0.001, nearest - time.monotonic()))
            else:
                delay = None
            if delay is None:
                self._sweep_wakeup.clear()
                await self._sweep_wakeup.wait()
                continue
            self._sweep_wakeup.clear()
            try:
                await asyncio.wait_for(self._sweep_wakeup.wait(), delay)
            except asyncio.TimeoutError:
                pass

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await self._not_empty.wait()

            # Batch-formation window: the idle model waits at most
            # `batch_wait_s` for more work, up to `batch_size` images.
            self._handoff = []
            window_end = loop.time() + self.batch_wait_s
            while len(self._handoff) < self.batch_size:
                self._purge_expired_pending()
                while self._pending and len(self._handoff) < self.batch_size:
                    self._handoff.append(self._pending.popleft())
                if len(self._handoff) >= self.batch_size:
                    break
                remaining = window_end - loop.time()
                if remaining <= 0:
                    break
                self._not_empty.clear()
                try:
                    await asyncio.wait_for(self._not_empty.wait(), remaining)
                except asyncio.TimeoutError:
                    break

            # Final gate: cancelled/expired orders never enter the model. Frames
            # dropped here are settled explicitly -- leaving them in `queued`
            # would make them invisible to capacity/drain accounting (zombies)
            # and would violate "queued expiry settles without a query".
            now = time.monotonic()
            batch: list[Order] = []
            for o in self._handoff:
                if o.state != QUEUED:
                    continue  # cancelled inside the window: already settled
                if o.deadline_at <= now:
                    o.state = EXPIRED
                    continue
                batch.append(o)
            self._handoff = []
            if not batch:
                continue

            batch_id = uuid.uuid4().hex[:8]
            try:
                tensors = torch.stack([o.tensor for o in batch])
            except Exception:  # noqa: BLE001 - a bad tensor must still settle the batch
                log.exception("could not stack batch %s; settling as failed", batch_id)
                for o in batch:
                    o.state = FAILED
                continue

            for o in batch:
                o.state = PROCESSING
                o.batch_id = batch_id
            self._inflight = len(batch)
            self.batches_started += 1
            self.last_batch_size = len(batch)

            # The batch is committed: from here on the slot is held until the
            # compute returns, even if every frame in it is cancelled/expired.
            try:
                scores = await loop.run_in_executor(self._executor, self._infer, tensors)
            except asyncio.CancelledError:
                # Worker shutdown: do not leave phantom capacity behind.
                self._inflight = 0
                raise
            except Exception:  # noqa: BLE001 - model failure must not kill the worker
                # Settle the whole batch (failed), then keep serving later work.
                log.exception("inference failed for batch %s", batch_id)
                for o in batch:
                    if o.state == PROCESSING:
                        o.state = FAILED
                self._inflight = 0
                continue

            # Write-back must settle EVERY still-processing frame in the batch,
            # even if the model returns a malformed result for some of them;
            # otherwise those frames would hold a slot forever and drain would
            # never report ready.
            try:
                now = time.monotonic()
                for idx, o in enumerate(batch):
                    if o.state != PROCESSING:
                        # Cancelled or expired mid-flight: result is discarded
                        # and the late result must NOT revive the order.
                        continue
                    if o.deadline_at <= now:
                        o.state = EXPIRED
                        continue
                    if idx >= len(scores):
                        o.state = FAILED
                        continue
                    sc = scores[idx]
                    predicted = max(range(len(sc)), key=lambda i: sc[i])
                    o.state = COMPLETED
                    o.result = {"predicted_class": predicted, "scores": sc}
            finally:
                for o in batch:
                    if o.state == PROCESSING:
                        o.state = FAILED
                self._inflight = 0

    # ------------------------------------------------------------------ #
    # runs on the single executor thread — never touches shared state
    # ------------------------------------------------------------------ #
    def _infer(self, batch_tensor: torch.Tensor) -> list[list[float]]:
        if self.inference_delay_s > 0:
            time.sleep(self.inference_delay_s)  # demo: simulate a slow model
        with torch.inference_mode():
            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
        return [[float(v) for v in row] for row in probs]
