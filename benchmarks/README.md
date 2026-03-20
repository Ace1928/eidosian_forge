# Benchmarks

## Model Setup
Use the curated model catalog in `config/model_catalog.json` and the downloader script:

```bash
./eidosian_venv/bin/python scripts/download_local_models.py --profile core
./eidosian_venv/bin/python scripts/download_local_models.py --profile toolcalling
./eidosian_venv/bin/python scripts/download_local_models.py --profile multimodal
```

Fast GraphRAG-only setup:

```bash
./scripts/download_graphrag_models.sh
```

Canonical selected runtime models are tracked in:

- `config/model_selection.json`
- `docs/MODEL_SELECTION.md`

## GraphRAG Runtime Benchmark
Use `run_graphrag_bench.py` to run end-to-end GraphRAG indexing + query against local `llama.cpp` servers with strict output quality gates.

### Run
```bash
./eidosian_venv/bin/python benchmarks/run_graphrag_bench.py
```

### Strict quality gates
- Placeholder community reports are blocked (`EIDOS_GRAPHRAG_ALLOW_PLACEHOLDER` must stay disabled).
- Community reports are validated for schema completeness and entity relevance.
- Non-informative query answers fail the run.
- Benchmark exits non-zero on any failed stage.

### Key env toggles
- `EIDOS_GRAPHRAG_QUERY_METHOD` (default `global`)
- `EIDOS_GRAPHRAG_LLM_MODEL` / `EIDOS_GRAPHRAG_LLM_MODEL_FALLBACK`
- `EIDOS_GRAPHRAG_MAX_COMPLETION_TOKENS` / `EIDOS_GRAPHRAG_MAX_TOKENS`
- `EIDOS_LLAMA_CTX_SIZE` / `EIDOS_LLAMA_PARALLEL` / `EIDOS_LLAMA_TEMPERATURE`
- `EIDOS_MODEL_SELECTION_PATH` (default `config/model_selection.json`)

### With profiler
```bash
./eidosian_venv/bin/python benchmarks/run_graphrag_bench.py --profile
```

### Outputs
- `reports/graphrag/bench_metrics_*.json`: index/query latency and query answer snapshot.
- `reports/graphrag/bench_profile_*.prof`: cProfile binary profile.
- `reports/graphrag/bench_profile_*.txt`: top cumulative profile functions.
- `data/graphrag_test/workspace/output/*`: GraphRAG parquet/json artifacts.

## Federated Qualitative Assessment
Use `graphrag_qualitative_assessor.py` to score GraphRAG artifacts with deterministic measurement contracts plus multi-model judge consensus.

### Run
```bash
./eidosian_venv/bin/python benchmarks/graphrag_qualitative_assessor.py \
  --workspace-root data/graphrag_test/workspace \
  --report-dir reports/graphrag
```

### Deterministic-only mode
```bash
./eidosian_venv/bin/python benchmarks/graphrag_qualitative_assessor.py \
  --workspace-root data/graphrag_test/workspace \
  --report-dir reports/graphrag \
  --metrics-json reports/graphrag/bench_metrics_<stamp>.json \
  --skip-judges
```

### Deterministic contract dimensions
- `pipeline_integrity`
- `workflow_completeness`
- `entity_coverage`
- `relationship_density`
- `community_report_quality`
- `query_answer_quality`
- `runtime_score`

### Judge dimensions
- `factuality`
- `grounding`
- `coherence`
- `usefulness`
- `risk_awareness`

### Outputs
- `reports/graphrag/qualitative_assessment_*.json`
- `reports/graphrag/qualitative_assessment_*.md`

Schema reference:
- `benchmarks/schemas/graphrag_qualitative_assessment.schema.json`

Judge defaults are loaded from `config/model_selection.json` (falling back to Qwen 0.5B + Llama 1B if absent).

## Multi-Domain Model Suite
Use `model_domain_suite.py` to benchmark local or Ollama-hosted models across tool calling, extraction, reasoning, ambiguity handling, coding, safety, and optional multimodal OCR. Ollama runs can sweep explicit thinking modes such as `off` and `on`.

### Run (catalog-driven)
```bash
./eidosian_venv/bin/python benchmarks/model_domain_suite.py --profile toolcalling
```

### Run with explicit models
```bash
./eidosian_venv/bin/python benchmarks/model_domain_suite.py \
  --model qwen_0_5b=models/Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  --model arch_3b=models/Arch-Function-3B-Q6_K.gguf
```

### Run an Ollama reasoning sweep
```bash
./eidosian_venv/bin/python benchmarks/model_domain_suite.py \
  --ollama-model qwen35=qwen3.5:2b \
  --thinking-mode off \
  --thinking-mode on
```

### Outputs
- `reports/model_domain_suite/model_domain_suite_*.json`
- `reports/model_domain_suite/model_domain_suite_*.md`
- `reports/model_domain_suite/model_domain_suite_latest.json`
- `reports/model_domain_suite/model_domain_suite_latest.md`

## Entity Proof Scorecard
Use `scripts/entity_proof_suite.py` to aggregate the latest benchmark, continuity, governance, runtime, and red-team evidence into a single externally legible proof bundle.

### Run
```bash
./eidosian_venv/bin/python scripts/entity_proof_suite.py
```

### What it aggregates
- model-domain benchmark evidence
- GraphRAG benchmark and qualitative assessment evidence
- consciousness benchmark / trial / RAC-AP validation evidence
- Linux parity evidence
- runtime coordinator, scheduler, local-agent, and documentation drift state
- governance surface checks for self-modification and rollback/red-team gating
- freshness policy for stale or missing evidence
- regression deltas against the previous scorecard
- imported external benchmark evidence under `reports/external_benchmarks/<suite>/latest.json`
- migration/replay reproducibility evidence from `scripts/migration_replay_scorecard.py`
- session-bridge continuity/import evidence from `data/runtime/session_bridge/*.json`
- recent proof-history deltas from prior `reports/proof/entity_proof_scorecard_*.json` artifacts

### Outputs
- `reports/proof/entity_proof_scorecard_<stamp>.json`
- `reports/proof/entity_proof_scorecard_<stamp>.md`
- `reports/proof/entity_proof_scorecard_latest.json`
- `reports/proof/entity_proof_scorecard_latest.md`

### Current purpose
- make missing evidence explicit rather than implicit
- provide one reportable scorecard for external review
- turn internal benchmark fragments into a publishable proof surface
- include identity continuity trend/delta history, not just a single-point score
- publish external benchmark result tables directly in the proof markdown, not only JSON artifacts
- publish recent proof-history trend rows directly in the proof markdown
- keep cross-interface continuity evidence visible in the proof bundle and manifest

## External Benchmark Import
Use `scripts/import_external_benchmark.py` to normalize upstream external-suite results into the proof pipeline.

### Run
```bash
./eidosian_venv/bin/python scripts/import_external_benchmark.py \
  --suite agentbench \
  --input path/to/upstream_summary.json \
  --source-url https://github.com/THUDM/AgentBench
```

### Outputs
- `reports/external_benchmarks/<suite>/<suite>_<stamp>.json`
- `reports/external_benchmarks/<suite>/latest.json`

## AgencyBench Reference Import
Use `scripts/import_agencybench_reference.py` to aggregate the official sample `claude/meta_eval.json` artifacts from an `AgencyBench` checkout into a bounded external-benchmark reference bundle.

### Run
```bash
./eidosian_venv/bin/python scripts/import_agencybench_reference.py \
  --agencybench-root /path/to/AgencyBench
```

### Outputs
- `reports/external_benchmarks/agencybench/agencybench_<stamp>.json`
- `reports/external_benchmarks/agencybench/latest.json`

## AgentBench Reference Import
Use `scripts/import_agentbench_reference.py` to normalize the official published AgentBench leaderboard CSV into the proof pipeline as a second mainstream external benchmark reference.

### Run
```bash
./eidosian_venv/bin/python scripts/import_agentbench_reference.py \
  --csv docs/external_references/2026-03-20-agentbench/AgentBench-leaderboard.csv
```

### Outputs
- `reports/external_benchmarks/agentbench/agentbench_<stamp>.json`
- `reports/external_benchmarks/agentbench/latest.json`

## AgencyBench Eidos Live Runs
Use `scripts/run_agencybench_eidos.py` to generate real Eidos-run benchmark artifacts against selected official AgencyBench scenarios.

### Scenario 2: Filesystem Workflow
This scenario uses the official AgencyBench MCP `scenario2` workspace contract. Two engines are supported:
- `local_agent`
- `deterministic`

Deterministic run:
```bash
./eidosian_venv/bin/python scripts/run_agencybench_eidos.py \
  --scenario scenario2 \
  --engine deterministic \
  --agencybench-root /path/to/AgencyBench
```

Bounded local-agent run:
```bash
./eidosian_venv/bin/python scripts/run_agencybench_eidos.py \
  --scenario scenario2 \
  --engine local_agent \
  --model qwen3.5:2b \
  --agencybench-root /path/to/AgencyBench \
  --timeout-sec 1800 \
  --keep-alive 4h
```

Current live artifact:
- `reports/external_benchmarks/agencybench_eidos_scenario2_deterministic/latest.json`

Live runtime observability artifacts for non-deterministic runs:
- `data/runtime/external_benchmarks/agencybench/<scenario>/<stamp>/status.json`
- `data/runtime/external_benchmarks/agencybench/<scenario>/<stamp>/attempts.jsonl`
- `data/runtime/external_benchmarks/agencybench/<scenario>/<stamp>/model_trace.jsonl`

Atlas surfaces the latest live runtime benchmark state through:
- `GET /api/benchmarks/runtime`
- `POST /api/benchmarks/runtime/run`
- `GET /api/benchmarks/runtime/run/status`
- `GET /api/benchmarks/runtime/run/history`
- the `Runtime Benchmarks` table on the Atlas home page

Related operator evidence now exposed through Atlas:
- `POST /api/proof/refresh`
- `GET /api/proof/refresh/status`
- `GET /api/proof/refresh/history`
- `POST /api/runtime-artifacts/audit`
- `GET /api/runtime-artifacts/audit/status`
- `GET /api/runtime-artifacts/audit/history`

### Scenario 1: GitHub Workflow
This scenario uses the official AgencyBench MCP `scenario1` GitHub workflow contract and executes it against a disposable benchmark repository under the authenticated GitHub account.

Run:
```bash
./eidosian_venv/bin/python scripts/run_agencybench_eidos.py \
  --scenario scenario1 \
  --engine deterministic \
  --repo-visibility private
```

Requirements:
- authenticated `gh`
- repo-scoped GitHub token
- outbound GitHub API access

Current live artifact:
- `reports/external_benchmarks/agencybench_eidos_scenario1_deterministic/latest.json`

### Notes
- The deterministic runners are intended to prove end-to-end execution against the official scenario contracts.
- The `local_agent` scenario2 path exists, but the current `qwen3.5:2b` Ollama serving path remains unstable for long benchmark turns and still needs remediation before it is a trustworthy external-proof path.
- Scenario1 currently supports deterministic execution only.

### Outputs
- `reports/external_benchmarks/agencybench_eidos_scenario1_deterministic/latest.json`
- `reports/external_benchmarks/agencybench_eidos_scenario1_deterministic/latest_detailed.json`
- `reports/external_benchmarks/agencybench_eidos_scenario2_deterministic/latest.json`
- `reports/external_benchmarks/agencybench_eidos_scenario2_deterministic/latest_detailed.json`

## Migration Replay Scorecard
Use `scripts/migration_replay_scorecard.py` to report replay and portability evidence.

### Run
```bash
./eidosian_venv/bin/python scripts/migration_replay_scorecard.py
```

### Outputs
- `reports/proof/migration_replay_scorecard_<stamp>.json`
- `reports/proof/migration_replay_scorecard_<stamp>.md`
- `reports/proof/migration_replay_scorecard_latest.json`
- `reports/proof/migration_replay_scorecard_latest.md`
