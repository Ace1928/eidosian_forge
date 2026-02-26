# Model Selection (Canonical)

This is the active model contract for Doc Forge + GraphRAG.

- Canonical file: `config/model_selection.json`
- Last updated from sweep: `reports/graphrag_sweep/model_selection_20260226_084600.json`

## Active Models

- Completion (primary): `models/Qwen2.5-0.5B-Instruct-Q8_0.gguf`
- Completion (fallback): `models/Llama-3.2-1B-Instruct-Q8_0.gguf`
- Embedding (primary): `models/nomic-embed-text-v1.5.Q4_K_M.gguf`
- Judge set:
  - `qwen=models/Qwen2.5-0.5B-Instruct-Q8_0.gguf`
  - `llama=models/Llama-3.2-1B-Instruct-Q8_0.gguf`

## Selection Rationale

- Latest strict sweep winner: `qwen2_5_0_5b_instruct` (`score=0.8572`, `rank=A`, `bench_ok=true`).
- Full 9-model sweep with disabled timeout cutoffs confirmed:
  - higher-scoring valid alternatives were slower (`qwen2_5_3b_instruct`, `llama_3_2_3b_instruct`)
  - several models failed strict GraphRAG quality gates (`bench_ok=false`)
- Embedding standard remains `models/nomic-embed-text-v1.5.Q4_K_M.gguf`.

## Updating Selection

1. Run a full federated sweep:
   ```bash
   ./eidosian_venv/bin/python benchmarks/graphrag_model_sweep.py \
     --assessment-timeout 0 \
     --judge-start-timeout 0 \
     --judge-request-timeout 0
   ```
2. Verify `reports/graphrag_sweep/model_selection_latest.json`.
3. Update `config/model_selection.json` to match the validated winner.
4. Keep this document synchronized with `config/model_selection.json`.
