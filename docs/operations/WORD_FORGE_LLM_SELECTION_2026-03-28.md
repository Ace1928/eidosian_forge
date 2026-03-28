# Word Forge Local LLM Selection — 2026-03-28

## Decision

Promote `models/Qwen3.5-2B-IQ4_XS.gguf` to the default local Word Forge model.

## Why

Word Forge needs a local model that is small enough for Termux-class hardware but strong enough to reliably produce structured lexical outputs.

The smallest previously-integrated Qwen 3.5 candidate, `Qwen3.5-0.8B-Q4_K_M.gguf`, was fast and acceptable for short definitions, but it underperformed on the structured extraction tasks that matter most to Word Forge.

`Qwen3.5-2B-IQ4_XS.gguf` materially improved structured lexical extraction while keeping the memory footprint far below the larger 2B Q8 and 3B-class alternatives.

## Compared Models

- `models/Qwen3.5-0.8B-Q4_K_M.gguf`
- `models/Qwen3.5-2B-IQ4_XS.gguf`

## Benchmark Shape

The comparison used `word_forge/scripts/benchmark_llm_backends.py` across three small Word Forge-style tasks:

1. concise definition generation
2. structured term extraction JSON
3. structured G2P JSON

## Results

### `Qwen3.5-0.8B-Q4_K_M.gguf`

- overall score: `0.667`
- definition: passed
- term extraction: failed due to incomplete / mis-shaped structured output under the tested settings
- g2p: passed, but with weak phonetic fidelity

### `Qwen3.5-2B-IQ4_XS.gguf`

- overall score: `1.0`
- definition: passed
- term extraction: passed
- g2p: passed

## Operational Tradeoff

`Qwen3.5-2B-IQ4_XS.gguf` is slower than `0.8B`, but the quality gain is large enough to justify the promotion for production Word Forge work.

The `0.8B` model remains a useful fallback for lower-memory situations.

## Contract

Word Forge now prefers local models in this order:

1. `Qwen3.5-2B-IQ4_XS.gguf`
2. `Qwen3.5-2B-Q4_K_M.gguf`
3. `Qwen3.5-0.8B-Q4_K_M.gguf`
4. `Qwen3.5-0.8B-IQ4_XS.gguf`
5. `Qwen2.5-0.5B-Instruct-Q8_0.gguf`
6. fallback remote default: `qwen/qwen3.5-2b-instruct`

## Overrides

- `EIDOS_WORD_FORGE_LLM_MODEL`
- `EIDOS_WORD_FORGE_GGUF_MODEL`
- `EIDOS_WORD_FORGE_LLAMA_CLI`

## Next Improvements

- benchmark `Qwen3.5-2B-Q4_K_M.gguf` as a possible higher-quality replacement if latency remains tolerable
- add a Word Forge-specific structured-output repair pass for malformed JSON
- add benchmark artifact publication into Atlas / proof surfaces if Word Forge LLM selection becomes an operator-facing concern
