"""End-to-end tests for POST /drain and GET /drain.

Covers the hand-off contract with the deployment layer: after drain starts new
frames get 503, already accepted frames (including ones racing with the drain
call) settle under the normal batching/deadline rules, and the service only
reports safe-to-shutdown when the backlog AND the actual model compute are
both empty.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager

from app.config import Settings
from app.main import create_app

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
MODEL = Path(__file__).resolve().parent.parent / "app" / "model.pt"
SAMPLE_NAMES = ["sample_gradient.png", "sample_circles.png", "sample_stripes.png", "sample_noise.png"]


def make_settings(**kw) -> Settings:
    kw.setdefault("model_path", str(MODEL))
    return Settings(**kw)


class Client:
    def __init__(self, http: httpx.AsyncClient, app):
        self.http = http
        self.app = app

    async def submit(self, name: str, deadline_ms: int = 60_000):
        data = (SAMPLES / name).read_bytes()
        return await self.http.post(
            "/predict",
            files={"file": (name, data, "image/png")},
            data={"deadline_ms": str(deadline_ms)},
        )

    async def get(self, order_id: str):
        return (await self.http.get(f"/orders/{order_id}")).json()

    async def cancel(self, order_id: str):
        return await self.http.post(f"/orders/{order_id}/cancel")

    async def drain(self):
        return await self.http.post("/drain")

    async def drain_status(self):
        r = await self.http.get("/drain")
        assert r.status_code == 200, r.text
        return r.json()

    async def wait_state(self, order_id: str, states, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            body = await self.get(order_id)
            if body["status"] in states:
                return body
            await asyncio.sleep(0.01)
        raise AssertionError(f"order {order_id} never reached {states}: {body}")

    async def wait_drain(self, predicate, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        body = await self.drain_status()
        while time.monotonic() < deadline:
            body = await self.drain_status()
            if predicate(body):
                return body
            await asyncio.sleep(0.01)
        raise AssertionError(f"drain condition never met: {body}")


@pytest.fixture
async def make_client():
    clients = []

    async def factory(**settings_kw):
        app = create_app(make_settings(**settings_kw))
        cm = LifespanManager(app)
        await cm.__aenter__()
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
        client = Client(http, app)
        clients.append((client, cm))
        return client

    yield factory
    for client, cm in clients:
        await client.http.aclose()
        await cm.__aexit__(None, None, None)


# --------------------------------------------------------------------- #
# drain lifecycle
# --------------------------------------------------------------------- #
async def test_drain_reports_ready_when_idle_and_rejects_new_work(make_client):
    c = await make_client()
    before = await c.drain_status()
    assert before["draining"] is False
    assert before["ready_to_shutdown"] is False  # never claims ready before drain

    r = await c.drain()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["draining"] is True
    assert body["ready_to_shutdown"] is True
    assert body["unsettled_frames"] == 0
    assert body["model_computing"] is False

    rejected = await c.submit("sample_gradient.png")
    assert rejected.status_code == 503, rejected.text
    # Rejected work must not be accounted as accepted.
    assert (await c.drain_status())["accepted_orders"] == 0


async def test_drain_settles_accepted_frames_then_ready(make_client):
    c = await make_client(batch_wait_ms=30)
    ids = []
    for name in SAMPLE_NAMES:
        r = await c.submit(name)
        assert r.status_code == 201
        ids.append(r.json()["order_id"])

    first = (await c.drain()).json()
    assert first["draining"] is True
    assert first["accepted_orders"] == 4
    assert first["unsettled_frames"] > 0 or first["model_computing"]  # drain observed work
    assert first["ready_to_shutdown"] is False

    # Already accepted frames keep completing under the normal rules.
    for oid in ids:
        body = await c.wait_state(oid, {"completed"})
        assert body["result"]["scores"]

    final = await c.wait_drain(lambda b: b["ready_to_shutdown"])
    assert final["unsettled_frames"] == 0
    assert final["queued_frames"] == 0
    assert final["computing_frames"] == 0
    assert final["model_computing"] is False
    assert final["accepted_orders"] == 4


async def test_repeated_drain_does_not_reset_progress(make_client):
    c = await make_client(inference_delay_ms=700)
    r = await c.submit("sample_gradient.png")
    assert r.status_code == 201

    first = (await c.drain()).json()
    await asyncio.sleep(0.25)
    second = (await c.drain()).json()  # repeat POST
    third = (await c.drain()).json()

    assert first["draining"] and second["draining"] and third["draining"]
    assert second["accepted_orders"] == first["accepted_orders"] == 1
    assert second["drain_elapsed_ms"] >= first["drain_elapsed_ms"]
    assert third["drain_elapsed_ms"] >= second["drain_elapsed_ms"]
    # The repeat must not clear the backlog bookkeeping.
    assert second["unsettled_frames"] >= 1
    assert third["unsettled_frames"] >= 1
    assert third["ready_to_shutdown"] is False

    final = await c.wait_drain(lambda b: b["ready_to_shutdown"])
    assert final["accepted_orders"] == 1


async def test_query_and_cancel_remain_available_during_drain(make_client):
    c = await make_client(inference_delay_ms=800)
    ids = []
    for i in range(5):  # one batch of 4 computes, 5th stays queued
        r = await c.submit(SAMPLE_NAMES[i % 4])
        assert r.status_code == 201
        ids.append(r.json()["order_id"])
    await asyncio.sleep(0.15)

    assert (await c.drain()).json()["draining"] is True

    # query still works
    body = await c.get(ids[0])
    assert body["status"] in {"processing", "queued"}

    # cancel a still-queued frame during drain
    victim = ids[4]
    assert (await c.get(victim))["status"] == "queued"
    assert (await c.cancel(victim)).status_code == 200
    assert (await c.get(victim))["status"] == "cancelled"

    # cancel an in-flight frame during drain
    in_flight = ids[0]
    assert (await c.cancel(in_flight)).status_code == 200
    assert (await c.get(in_flight))["status"] == "cancelled"

    final = await c.wait_drain(lambda b: b["ready_to_shutdown"])
    assert final["unsettled_frames"] == 0
    assert (await c.get(victim))["result"] is None
    assert (await c.get(in_flight))["result"] is None
    for oid in ids[1:4]:
        assert (await c.get(oid))["status"] == "completed"


async def test_cancelled_inflight_frames_still_wait_for_compute_end(make_client):
    c = await make_client(inference_delay_ms=900)
    ids = []
    for name in SAMPLE_NAMES:  # exactly one batch
        r = await c.submit(name)
        ids.append(r.json()["order_id"])
    await asyncio.sleep(0.15)
    assert (await c.get(ids[0]))["status"] == "processing"

    for oid in ids:
        assert (await c.cancel(oid)).status_code == 200

    # All four are already marked cancelled, yet the model is still running:
    # this must NOT be reported as safe to shut down.
    status = (await c.drain()).json()
    assert status["model_computing"] is True
    assert status["computing_frames"] == 4
    assert status["unsettled_frames"] == 4
    assert status["ready_to_shutdown"] is False

    await asyncio.sleep(0.2)
    still = await c.drain_status()
    assert still["ready_to_shutdown"] is False

    final = await c.wait_drain(lambda b: b["ready_to_shutdown"])
    assert final["computing_frames"] == 0
    for oid in ids:
        body = await c.get(oid)
        assert body["status"] == "cancelled" and body["result"] is None


async def test_queued_expiry_settles_without_queries(make_client):
    c = await make_client(inference_delay_ms=900)
    # Occupy the model with a full batch so later frames stay queued.
    busy = []
    for name in SAMPLE_NAMES:
        r = await c.submit(name)
        busy.append(r.json()["order_id"])
    await asyncio.sleep(0.15)

    pending = []
    for _ in range(3):
        r = await c.submit("sample_gradient.png", deadline_ms=250)
        assert r.status_code == 201
        pending.append(r.json()["order_id"])

    assert (await c.drain()).json()["draining"] is True
    engine = c.app.state.engine
    # Poll ONLY the drain endpoint; never query the individual orders.
    await c.drain_status()
    await asyncio.sleep(0.35)  # deadlines pass while still queued

    # Read the engine directly (no HTTP query at all): the background sweeper
    # must already have settled the queued frames.
    await asyncio.sleep(0.1)
    assert engine._unsettled() == engine._inflight, (
        "queued expiry must settle without any query"
    )

    status = await c.drain_status()
    # The queued frames are settled by the engine itself (timer sweep), so
    # drain progress does not depend on anyone querying /orders.
    assert status["queued_frames"] == 0
    assert status["unsettled_frames"] == status["computing_frames"]
    assert status["ready_to_shutdown"] is False  # first batch still computing

    final = await c.wait_drain(lambda b: b["ready_to_shutdown"])
    assert final["unsettled_frames"] == 0
    assert final["model_computing"] is False

    for oid in pending:
        body = await c.get(oid)
        assert body["status"] == "expired"
        assert body["result"] is None
        assert body["batch_id"] is None  # never entered the model
    for oid in busy:
        assert (await c.get(oid))["status"] == "completed"


# --------------------------------------------------------------------- #
# submit/drain interleaving
# --------------------------------------------------------------------- #
async def test_interleaved_submits_and_drain_count_every_201(make_client):
    c = await make_client(inference_delay_ms=250, batch_wait_ms=10)

    async def fire(i: int):
        # Spread submissions across event-loop turns so some land before the
        # drain flag and some hit it afterwards.
        await asyncio.sleep(i * 0.001)
        return await c.submit(SAMPLE_NAMES[i % 4])

    async def drain_soon():
        await asyncio.sleep(0.012)
        return await c.drain()

    results = await asyncio.gather(*[fire(i) for i in range(14)], drain_soon())
    drain_resp = results[-1]
    submit_results = results[:-1]

    accepted = [r.json()["order_id"] for r in submit_results if r.status_code == 201]
    # Anything not accepted is explicitly rejected, never silently dropped.
    for r in submit_results:
        assert r.status_code in {201, 503, 429}, r.text
    assert drain_resp.status_code == 200

    engine = c.app.state.engine
    status = await c.drain_status()
    assert status["accepted_orders"] == len(accepted)
    assert engine.accepted_orders == len(accepted)
    # Once draining is visible, a fresh submit is always rejected.
    assert (await c.submit("sample_gradient.png")).status_code == 503

    final = await c.wait_drain(lambda b: b["ready_to_shutdown"])
    assert final["unsettled_frames"] == 0
    assert final["model_computing"] is False
    assert final["accepted_orders"] == len(accepted)
    # Every order that got a 201 was settled through the drain path.
    for oid in accepted:
        body = await c.get(oid)
        assert body["status"] in {"completed", "expired", "cancelled", "failed"}
        assert body["status"] != "queued" and body["status"] != "processing"


# --------------------------------------------------------------------- #
# inference failure
# --------------------------------------------------------------------- #
class _FlakyModel:
    """TorchScript-compatible wrapper that fails while `fail` is set."""

    def __init__(self, real):
        self.real = real
        self.fail = True

    def eval(self):
        return self

    def __call__(self, x):
        if self.fail:
            raise RuntimeError("simulated model failure")
        return self.real(x)


async def test_inference_failure_settles_batch_and_later_frames_progress(make_client):
    c = await make_client(batch_wait_ms=10)
    engine = c.app.state.engine
    engine.model = _FlakyModel(engine.model)

    first = []
    for name in SAMPLE_NAMES:
        r = await c.submit(name)
        assert r.status_code == 201
        first.append(r.json()["order_id"])
    for oid in first:
        await c.wait_state(oid, {"failed"})

    # The failed batch released its capacity and did not stall the worker.
    status = await c.drain_status()
    assert status["computing_frames"] == 0
    assert status["queued_frames"] == 0

    engine.model.fail = False
    later = []
    for name in SAMPLE_NAMES[:2]:
        r = await c.submit(name)
        assert r.status_code == 201
        later.append(r.json()["order_id"])
    for oid in later:
        body = await c.wait_state(oid, {"completed"})
        assert body["result"]["scores"]

    # Drain with a mixture of failed and completed accepted frames.
    assert (await c.drain()).json()["draining"] is True
    final = await c.wait_drain(lambda b: b["ready_to_shutdown"])
    assert final["accepted_orders"] == 6
    assert final["unsettled_frames"] == 0
