# `llm_forge/legacy_imports/development_head_2026-02-14_174300/oumi-main/src/oumi/inference`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:19:22Z`
- Path: `llm_forge/legacy_imports/development_head_2026-02-14_174300/oumi-main/src/oumi/inference`

## What It Is

Managed directory documentation for `llm_forge/legacy_imports/development_head_2026-02-14_174300/oumi-main/src/oumi/inference`, generated from tracked files and directory structure.

## Why It Exists

This directory exists to hold the `forge` surface for `llm_forge` and keep that responsibility separate from adjacent forge concerns.

## How It Works

- Tracked files in scope: `14`
- Child directories: `0`
- Tests detected: `False`
- Python modules: `anthropic_inference_engine, deepseek_inference_engine, gcp_inference_engine, gemini_inference_engine, llama_cpp_inference_engine, native_text_inference_engine, openai_inference_engine, parasail_inference_engine, remote_inference_engine, remote_vllm_inference_engine`

## Contents

- No managed child directories detected.

## Prominent Files

- [`__init__.py`](./__init__.py)
- [`anthropic_inference_engine.py`](./anthropic_inference_engine.py)
- [`deepseek_inference_engine.py`](./deepseek_inference_engine.py)
- [`gcp_inference_engine.py`](./gcp_inference_engine.py)
- [`gemini_inference_engine.py`](./gemini_inference_engine.py)
- [`llama_cpp_inference_engine.py`](./llama_cpp_inference_engine.py)
- [`native_text_inference_engine.py`](./native_text_inference_engine.py)
- [`openai_inference_engine.py`](./openai_inference_engine.py)

## Strengths

- The directory exposes importable Python modules rather than only opaque assets.

## Weaknesses / Risks

- No directly associated test coverage was detected under the tracked file set for this directory.
- The directory has many tracked files but no child-directory decomposition, which may make ownership blur over time.

## Next Steps

- Add focused tests or point this directory explicitly at its validating test surface.
- Keep this README synchronized with code and test changes through the managed documentation toolchain.

## References

- Parent README: [`llm_forge/legacy_imports/development_head_2026-02-14_174300/oumi-main/src/oumi/README.md`](../README.md)
- Forge README: [`llm_forge/README.md`](../../../../../../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
