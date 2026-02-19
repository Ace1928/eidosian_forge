# Local Model Bench (Termux/Linux)

Date: 2026-02-19 (UTC)

## Scope
This run selected and validated local models up to 3B parameters (plus embedding model) with:
- GraphRAG end-to-end benchmark + federated qualitative assessment.
- Multi-domain capability suite (tool calling, extraction, reasoning, coding, safety, plus multimodal OCR where available).

## Downloaded Models
All models were fetched via `scripts/download_local_models.py` using `config/model_catalog.json`.

- `models/Qwen2.5-0.5B-Instruct-Q8_0.gguf`
- `models/Llama-3.2-1B-Instruct-Q8_0.gguf`
- `models/Qwen2.5-1.5B-Instruct-Q8_0.gguf`
- `models/Qwen2.5-3B-Instruct-Q6_K.gguf`
- `models/Arch-Function-3B-Q6_K.gguf`
- `models/Qwen2.5-Coder-3B-Instruct-Q6_K.gguf`
- `models/Llama-3.2-3B-Instruct-Q6_K.gguf`
- `models/SmolLM2-1.7B-Instruct-Q6_K.gguf`
- `models/Qwen2-VL-2B-Instruct-Q4_K_M.gguf`
- `models/mmproj-Qwen2-VL-2B-Instruct-f16.gguf`
- `models/nomic-embed-text-v1.5.Q4_K_M.gguf`

## Bench Commands

```bash
./eidosian_venv/bin/python scripts/download_local_models.py --profile toolcalling
./eidosian_venv/bin/python scripts/download_local_models.py --profile multimodal
./eidosian_venv/bin/python benchmarks/graphrag_model_sweep.py --model qwen_0_5b=models/Qwen2.5-0.5B-Instruct-Q8_0.gguf --model llama_1b=models/Llama-3.2-1B-Instruct-Q8_0.gguf --model qwen_1_5b=models/Qwen2.5-1.5B-Instruct-Q8_0.gguf
./eidosian_venv/bin/python benchmarks/run_graphrag_bench.py --query "What is the relationship between Kael and Seraphina?"
./eidosian_venv/bin/python benchmarks/graphrag_qualitative_assessor.py --workspace-root data/graphrag_test/workspace --report-dir reports/graphrag --metrics-json reports/graphrag/bench_metrics_20260219_210522.json
./eidosian_venv/bin/python benchmarks/model_domain_suite.py --max-tokens 160 --model qwen_0_5b=models/Qwen2.5-0.5B-Instruct-Q8_0.gguf --model llama_1b=models/Llama-3.2-1B-Instruct-Q8_0.gguf --model qwen_1_5b=models/Qwen2.5-1.5B-Instruct-Q8_0.gguf --model smollm2_1_7b=models/SmolLM2-1.7B-Instruct-Q6_K.gguf
./eidosian_venv/bin/python benchmarks/model_domain_suite.py --profile multimodal --max-tokens 128
./eidosian_venv/bin/python benchmarks/model_domain_suite.py --skip-judges --max-tokens 96 --model qwen_3b=models/Qwen2.5-3B-Instruct-Q6_K.gguf
./eidosian_venv/bin/python benchmarks/model_domain_suite.py --skip-judges --max-tokens 96 --model arch_3b=models/Arch-Function-3B-Q6_K.gguf
./eidosian_venv/bin/python benchmarks/model_domain_suite.py --skip-judges --max-tokens 96 --model qwen_coder_3b=models/Qwen2.5-Coder-3B-Instruct-Q6_K.gguf
```

## Results

### GraphRAG (quality-gated)
- `reports/graphrag/bench_metrics_20260219_210522.json`
- `reports/graphrag/qualitative_assessment_20260219_210603.json`

Qwen2.5-0.5B result:
- `index_seconds`: `38.67`
- `query_seconds`: `19.30`
- `query_output`: `The relationship between Kael and Seraphina is that they are lovers.`
- qualitative score: `0.9121` (`A`)

Qwen2.5-1.5B result:
- failed strict GraphRAG quality gate in `create_community_reports_text` (placeholder disabled).

### Multi-domain suite (with judges)
- `reports/model_domain_suite/model_domain_suite_20260219_110238.json`

Ranking:
- `qwen_1_5b`: final `0.6250` (`C`)
- `llama_1b`: final `0.5917` (`C`)
- `smollm2_1_7b`: final `0.5125` (`D`)
- `qwen_0_5b`: final `0.4900` (`D`)

### Multimodal suite (with judges)
- `reports/model_domain_suite/model_domain_suite_20260219_110357.json`

Ranking:
- `qwen2_vl_2b_instruct_q4`: final `0.8057` (`B`)
- includes `vision_ocr: 1.0`

### Single-model deep checks (deterministic only)
- `reports/model_domain_suite/model_domain_suite_20260219_111114.json`
- `reports/model_domain_suite/model_domain_suite_20260219_111252.json`
- `reports/model_domain_suite/model_domain_suite_20260219_111412.json`

Scores:
- `qwen_3b`: final `1.0000` (`A`) on deterministic suite.
- `qwen_coder_3b`: final `0.7667` (`B`) on deterministic suite.
- `arch_3b`: final `0.6667` (`C`) on deterministic suite.

## Recommended Defaults

- GraphRAG default (current best validated quality gate):
  - completion: `models/Qwen2.5-0.5B-Instruct-Q8_0.gguf`
  - embedding: `models/nomic-embed-text-v1.5.Q4_K_M.gguf`

- Tool-calling default (balanced judged run):
  - `models/Qwen2.5-1.5B-Instruct-Q8_0.gguf`

- Tool-calling max quality (deterministic suite, slower):
  - `models/Qwen2.5-3B-Instruct-Q6_K.gguf`

- Coding/tool hybrid:
  - `models/Qwen2.5-Coder-3B-Instruct-Q6_K.gguf`

- Multimodal default:
  - `models/Qwen2-VL-2B-Instruct-Q4_K_M.gguf`
  - `models/mmproj-Qwen2-VL-2B-Instruct-f16.gguf`

## Upstream References
- Qwen2.5 release: https://qwenlm.github.io/blog/qwen2.5/
- Qwen2-VL release: https://qwenlm.github.io/blog/qwen2-vl/
- Meta Llama 3.2 model card and tool-use benchmarks: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md
- Arch-Function model card (function-calling focus): https://huggingface.co/katanemo/Arch-Function-3B
- llama.cpp multimodal tooling (mtmd): https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd
