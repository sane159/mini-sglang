# Mini-SGLang 调度层改造与混合负载压测

本仓库基于官方 [Mini-SGLang](https://github.com/sgl-project/mini-sglang) 进行二次开发，针对**长短 Prompt 混合负载下 Prefill 抢占导致在线 Decode 尾延迟飙升**的问题，对调度决策层进行了解耦与策略扩展，并在单卡 RTX 4090（Qwen3-4B）上完成了回归测试与消融实验。

## 核心改动

1. **调度决策层解耦与多策略支持**
   - 新增独立决策模块 `python/minisgl/scheduler/policy.py`（`SchedulePolicy`），将批次调度决策从 `Scheduler._schedule_next_batch` 中抽离，与底层 KV Cache / Chunked Prefill 逻辑解耦。
   - 扩展 `python/minisgl/scheduler/config.py` 与 `python/minisgl/server/args.py`，支持通过 CLI 动态切换三种策略：
     - `prefill_first`：默认策略，优先调度 Prefill。
     - `decode_first`：优先保障在线 Decode 批次。
     - `balanced`：引入**连续 Decode 步数上限（`--max-consecutive-decode`）**与 **Prefill Token 预算控制（`--balanced-prefill-budget`）**，在长请求切块与在线 Decode 之间交替调度；同时在 KV Cache / Table 容量不足时保留向 Decode 的安全回退机制。
2. **无 GPU 依赖的单元测试**
   - 新增 `tests/test_schedule_policy.py`（5 组用例），覆盖三种策略优先级、连续 Decode 边界切换、Token 预算截断、KV Cache 不足回退及非法参数校验。
3. **长短混合负载流式压测与消融框架**
   - 新增 `scripts/bench_mixed.py` 与 `scripts/run_experiments.sh`，构造带唯一前缀（防 RadixCache 命中）的 8K / 512-token 混合流式请求，精确采集 TTFT P95、ITL P95/P99/Max 与输出吞吐。

## 快速复现

```bash
# 1. 运行调度策略单元测试
python -m unittest discover -s tests -v

# 2. 启动 balanced 调度策略服务
python -m minisgl \
  --model /path/to/Qwen3-4B \
  --port 1919 \
  --max-running-requests 32 \
  --schedule-policy balanced \
  --max-consecutive-decode 4 \
  --balanced-prefill-budget 1024

# 3. 一键运行 5 组策略对比与消融实验
bash scripts/run_experiments.sh
