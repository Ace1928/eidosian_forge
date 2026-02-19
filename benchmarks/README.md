# Benchmarks

## GraphRAG Runtime Benchmark
Use `run_graphrag_bench.py` to run end-to-end GraphRAG indexing + global query against local `llama.cpp` servers.

### Run
```bash
./eidosian_venv/bin/python benchmarks/run_graphrag_bench.py
```

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
