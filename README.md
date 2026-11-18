# Micro-batch Image Classifier

为视觉部署场景（多台相机）提供的**限时图片分类**服务：FastAPI + CPU 版 PyTorch，
单进程微批（micro-batch）推理。提交图片时给出从受理起算的截止毫秒数；
超时或被取消的帧不会产出结果。

## 快速开始

```bash
docker compose up --build
# 服务: http://localhost:8000   交互文档: http://localhost:8000/docs
```

本地开发（无需 Docker）：

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/uvicorn app.main:app --port 8000
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/predict` | 提交图片（multipart `file`）+ 表单 `deadline_ms`（从受理起算）。返回 `order_id`。满额返回 **429** |
| GET | `/orders/{order_id}` | 按单号查询状态与结果 |
| POST | `/orders/{order_id}/cancel` | 按单号取消；已完成返回 **409** |
| POST | `/drain` | 摘流：拒绝新提交（503），开始排空已受理帧；重复调用不重置进度 |
| GET | `/drain` | 排空进度与可下线信号（未结清帧数、实际计算占用） |
| GET | `/stats` | 容量/状态计数（演示与观测用） |
| GET | `/health` | 存活探针 |

示例：

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@samples/sample_gradient.png" -F "deadline_ms=5000"
# {"order_id":"7f6770829f36","status":"queued",...}

curl http://localhost:8000/orders/7f6770829f36
# {"status":"completed","result":{"predicted_class":1,"scores":[...10 个分数...]},...}

curl -X POST http://localhost:8000/orders/7f6770829f36/cancel
```

单号状态机：`queued → processing → completed`；`queued/processing → cancelled|expired`。
`result.scores` 为 10 类 softmax 分数，与提交单号一一对应。

## 摘流与排空（POST /drain、GET /drain）

相机下线前调用 `POST /drain`，用 `GET /drain` 轮询可信的下线信号：

```bash
curl -X POST http://localhost:8000/drain
# {"draining":true,"ready_to_shutdown":false,"unsettled_frames":5,
#  "queued_frames":1,"computing_frames":4,"model_computing":true,
#  "accepted_orders":5,"settled_frames":0,...}

curl http://localhost:8000/drain
# 全部结清且无实际计算后：
# {"draining":true,"ready_to_shutdown":true,"unsettled_frames":0,
#  "queued_frames":0,"computing_frames":0,"model_computing":false,...}
```

语义：

- **新提交返回 503**：摘流后 `POST /predict` 一律 503；摘流前已返回 **201** 的单
  必定计入排空（受理与摘流标记在同一事件循环同步段内完成，不存在“已受理但未计入”）。
- **重复摘流不重置进度**：`POST /drain` 幂等，不重置计时、受理计数与积压统计。
- **已受理帧按原规则处理**：继续按原微批与截止规则完成；队列中到期的帧由引擎定时清理
  直接结清（**无需任何查询**），且不进入模型。
- **取消/到期帧与容量**：入批后取消或到期的帧仍占用容量，**必须等该批计算实际结束**
  才结清；`model_computing` 反映真实计算占用，而非仅状态标记。
- **查询与取消保持可用**：排空期间 `GET /orders/{id}`、`POST /orders/{id}/cancel` 正常运行。
- **可下线条件**：`ready_to_shutdown = true` 当且仅当已摘流、`unsettled_frames == 0`
  且 `model_computing == false`（即无排队帧、无入批帧、无实际模型计算）。
- **推理失败**：整批结清为 `failed`（释放容量），worker 继续处理后续已受理帧，不阻断排空。

`GET /drain` 在未摘流时也可读，此时 `draining`/`ready_to_shutdown` 均为 `false`。

## 规则实现（与需求逐条对应）

- **容量 16**：排队中 + 计算中合计最多 16 张，满额提交返回 429。
- **单批 ≤ 4，等批 ≤ 50ms**：模型空闲时最多等 50ms 凑批，凑满 4 张立即开批。
- **最多一批实际推理**：单个 worker 协程 + 单线程执行器，任何时刻只有一批在做前向计算。
- **排队取消/到期**：从队列移除、释放容量，**不进入模型**（`batch_id` 始终为空）。
  排队到期由引擎的定时清理任务结清，**无需任何查询**。
- **入批后取消/到期**：立即标记（查询即刻可见）、不供结果；**容量占用到该批计算实际结束**，
  期间不会因此另开一批（worker 严格串行）。
- **批内其他图片正常完成**，分数对应原单；批推理与逐张推理误差 ≤ 1e-5（实测约 7.5e-9）。
- **迟到结果不恢复取消/过期单**：结果回写只针对仍处于 `processing` 且未过期的单。
- **慢推理不挡 API**：阻塞式前向计算放在独立线程，事件循环始终可用。

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `MAX_CAPACITY` | 16 | 排队+计算中最大图片数 |
| `BATCH_SIZE` | 4 | 单批最大图片数 |
| `BATCH_WAIT_MS` | 50 | 空闲模型凑批等待上限 |
| `INFERENCE_DELAY_MS` | 0 | **演示用**：在推理线程内注入的人为延迟（`docker-compose.yml` 演示时设为 2000） |
| `MODEL_PATH` | `app/model.pt` | TorchScript 模型路径 |

## 用 Docker Compose 演示摘流

`docker-compose.yml` 默认带 `INFERENCE_DELAY_MS=2000`，便于在 Swagger UI
（`http://localhost:8000/docs`）里肉眼观察“计算中 → 结清”的过程：

```bash
docker compose up --build -d

# 1) 在 /docs 里连续 4 次调用 POST /predict（同一批，约 2s 推理）
# 2) 立即调用 POST /drain  → draining=true, model_computing=true,
#    ready_to_shutdown=false
# 3) 再调用 POST /predict  → 503
# 4) 轮询 GET /drain       → 约 2s 后 ready_to_shutdown=true
# 5) GET /orders/{id} 仍可查到对应 completed 结果

docker compose down
```

想恢复零延迟默认值，把 `docker-compose.yml` 中 `INFERENCE_DELAY_MS` 改回 `"0"`。

## 模型与样图

- `app/model.pt`：固定权重 TorchScript 模型（小型 CNN，10 类，输入 64×64 RGB）。
  由 `scripts/make_model.py` 以固定种子生成，权重 SHA-256 指纹见脚本输出，可复现。
- `samples/`：4 张确定性生成的样图（渐变/圆环/条纹/噪声）。

重新生成：

```bash
python scripts/make_model.py --out app/model.pt
python scripts/make_samples.py --out samples
```

## 测试

```bash
.venv/bin/pytest tests/ -q
```

覆盖：单张完成、批/单张分数一致性（≤1e-5）、凑批窗口、容量 429、排队取消、
入批取消（立即标记+容量仍占用+不另开批）、完成单取消冲突、排队过期、
计算中过期、迟到结果不复活、慢推理下 API 延迟。

`tests/test_drain.py` 另外覆盖：摘流后 503 与可下线信号、重复摘流不重置进度、
已受理帧完成、排空期间查询/取消、入批取消帧等计算结束、排队到期无需查询即结清、
提交与摘流交错时所有 201 均计入排空、推理失败结清本批且不阻断后续帧。

## 已知限制

- 订单记录保存在内存中，重启即丢失；未做持久化与历史订单清理（演示规模）。
- 单进程单 worker；多副本扩展需外加负载均衡与共享状态层，不在本交付范围。
