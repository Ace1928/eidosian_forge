# Part 11: Benchmarking and Self-Improvement

## Goal

Provide objective, repeatable measurements for capability growth, runtime performance, and regression control.

## Implemented

1. Internal benchmark suite:
- `agent_forge/src/agent_forge/consciousness/benchmarks.py`
- Measures:
  - tick latency (`p50`, `p95`, `max`)
  - events emitted per tick
  - workspace coherence
  - response complexity (`RCI`)
  - agency, boundary stability
  - world prediction error
  - report groundedness
  - meta confidence

2. Composite score and gate checks:
- Capability/performance weighted composite.
- Baseline delta comparison (`non_regression_vs_baseline`).
- Runtime gates:
  - `world_model_online`
  - `meta_online`
  - `report_online`
  - `latency_p95_under_100ms`

3. CLI and MCP integration:
- `eidctl consciousness benchmark`
- `eidctl consciousness latest-benchmark`
- `eidctl consciousness full-benchmark`
- `eidctl consciousness latest-full-benchmark`
- MCP tools:
  - `consciousness_kernel_benchmark`
  - `consciousness_kernel_latest_benchmark`
  - `consciousness_kernel_full_benchmark`
  - `consciousness_kernel_latest_full_benchmark`
- MCP resource:
  - `eidos://consciousness/runtime-latest-benchmark`
  - `eidos://consciousness/runtime-latest-full-benchmark`

4. External benchmark score ingestion:
- Optional score ingestion for:
  - `mmlu`
  - `gpqa`
  - `swe_bench_verified`
  - `human_eval`
- Normalized against documented targets and included in composite score.

## Usage

```bash
PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src \
./eidosian_venv/bin/python agent_forge/bin/eidctl consciousness benchmark \
  --dir state \
  --ticks 12 \
  --external-score mmlu=0.72 \
  --external-source mmlu=https://arxiv.org/abs/2009.03300 \
  --json
```

```bash
PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src \
./eidosian_venv/bin/python agent_forge/bin/eidctl consciousness full-benchmark \
  --dir state \
  --rounds 1 \
  --bench-ticks 2 \
  --trial-ticks 1 \
  --skip-llm \
  --skip-mcp \
  --json
```

## Acceptance Criteria

1. Benchmark run emits a `benchmark.run` event with score and gate payload.
2. Benchmarks persist to `reports/consciousness_benchmarks/`.
3. Latest benchmark can be retrieved via CLI and MCP.
4. Baseline delta and non-regression gate are populated when baseline is available.
5. Integrated benchmark persists to `reports/consciousness_integrated_benchmarks/` and emits `benchmark.integrated`.
