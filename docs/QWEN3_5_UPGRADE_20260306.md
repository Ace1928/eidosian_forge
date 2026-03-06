# Qwen3.5 Upgrade Notes

Date: 2026-03-06

## Purpose

Move Eidosian runtime defaults toward the current small Qwen release while keeping the stack live and configurable.

## Official References

- Alibaba Cloud press release:
  - https://www.alibabacloud.com/blog/alibaba-cloud-launches-qwen3-5-new-hybrid-reasoning-models-open-source-for-agentic-ai-development_603149
- Official Hugging Face collection:
  - https://huggingface.co/collections/Qwen/qwen35-68bb7bd4ea52fd8662eec8e0
- Official Ollama library tags:
  - https://ollama.com/library/qwen3.5

## Confirmed Availability

- Public Qwen3.5 release exists.
- Official small Ollama tags include:
  - `qwen3.5:2b`
  - `qwen3.5:2b-q4_K_M`
  - `qwen3.5:4b`
  - `qwen3.5:4b-q4_K_M`

## Local State Before Upgrade

- Installed Ollama models observed locally:
  - `nomic-embed-text:latest`
  - `qwen2.5:1.5b`
- Local GGUF inventory still centers on Qwen2.5 and Llama 3.2 artifacts under `models/`.

## Runtime Direction

- Promote Ollama runtime default to `qwen3.5:2b`
- Keep embeddings on `nomic-embed-text`
- Keep GGUF local generation on the best existing local artifact until a Qwen3.5 GGUF is actually present
- Normalize reasoning-aware responses so `thinking` is preserved as metadata and visible answer text is extracted consistently
- Use inference `thinking_mode: off` as the default operational policy for now to prevent reasoning-only token burn on short calls
- Re-evaluate `qwen3.5:4b` only after latency and quality data justify the larger footprint

## Follow-Up Work

1. Benchmark `qwen3.5:2b` across `thinking_mode=off|on` and judge the quality/latency tradeoff explicitly.
2. If quality and latency are acceptable, benchmark `qwen3.5:4b`.
3. Add or update local GGUF artifacts for Qwen3.5 before switching GGUF defaults.
4. Prune obsolete local models only after the replacement path is validated.

## Local Smoke Result

- `qwen3.5:2b` was pulled successfully into Ollama.
- On this local Termux stack, direct `/api/generate` and `/api/chat` initially appeared broken because the model emitted text into Ollama's `thinking` channel while visible answer fields stayed empty.
- Sending `think: false` produced normal visible completions immediately.
- The correct fix was response normalization plus an explicit default thinking policy, not reverting to `qwen2.5:1.5b`.
