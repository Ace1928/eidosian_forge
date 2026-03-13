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

### Outputs
- `reports/proof/entity_proof_scorecard_<stamp>.json`
- `reports/proof/entity_proof_scorecard_<stamp>.md`
- `reports/proof/entity_proof_scorecard_latest.json`
- `reports/proof/entity_proof_scorecard_latest.md`

### Current purpose
- make missing evidence explicit rather than implicit
- provide one reportable scorecard for external review
- turn internal benchmark fragments into a publishable proof surface

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
