# Benchmarks

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
