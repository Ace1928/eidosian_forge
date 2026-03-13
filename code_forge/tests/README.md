# `code_forge/tests`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:19:21Z`
- Path: `code_forge/tests`

## What It Is

Managed directory documentation for `code_forge/tests`, generated from tracked files and directory structure.

## Why It Exists

This directory exists to hold the `tests` surface for `code_forge` and keep that responsibility separate from adjacent forge concerns.

## How It Works

- Tracked files in scope: `22`
- Child directories: `0`
- Tests detected: `True`
- Python modules: `conftest, test_analyzer_granularity, test_bench_runner, test_canonicalize_planner, test_code, test_digester_drift, test_digester_pipeline, test_digester_schema, test_eval_os_otlp, test_eval_os_runner`

## Contents

- No managed child directories detected.

## Prominent Files

- [`conftest.py`](./conftest.py)
- [`test_analyzer_granularity.py`](./test_analyzer_granularity.py)
- [`test_bench_runner.py`](./test_bench_runner.py)
- [`test_canonicalize_planner.py`](./test_canonicalize_planner.py)
- [`test_code.py`](./test_code.py)
- [`test_digester_drift.py`](./test_digester_drift.py)
- [`test_digester_pipeline.py`](./test_digester_pipeline.py)
- [`test_digester_schema.py`](./test_digester_schema.py)

## Strengths

- A directly associated test surface is present in or below this directory.
- The directory exposes importable Python modules rather than only opaque assets.

## Weaknesses / Risks

- The directory has many tracked files but no child-directory decomposition, which may make ownership blur over time.

## Next Steps

- Keep this README synchronized with code and test changes through the managed documentation toolchain.

## References

- Parent README: [`code_forge/README.md`](../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
