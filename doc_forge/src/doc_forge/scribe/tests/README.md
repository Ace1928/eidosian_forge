# `doc_forge/src/doc_forge/scribe/tests`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:19:21Z`
- Path: `doc_forge/src/doc_forge/scribe/tests`

## What It Is

Managed directory documentation for `doc_forge/src/doc_forge/scribe/tests`, generated from tracked files and directory structure.

## Why It Exists

This directory exists to hold the `tests` surface for `doc_forge` and keep that responsibility separate from adjacent forge concerns.

## How It Works

- Tracked files in scope: `7`
- Child directories: `0`
- Tests detected: `True`
- Python modules: `conftest, test_config, test_extract, test_generate, test_judge, test_service, test_state`
- Detected API routes: `GET /health`

## Contents

- No managed child directories detected.

## Prominent Files

- [`conftest.py`](./conftest.py)
- [`test_config.py`](./test_config.py)
- [`test_extract.py`](./test_extract.py)
- [`test_generate.py`](./test_generate.py)
- [`test_judge.py`](./test_judge.py)
- [`test_service.py`](./test_service.py)
- [`test_state.py`](./test_state.py)

## API Surface

- `GET /health`

## Strengths

- A directly associated test surface is present in or below this directory.
- The directory contains detected HTTP/API route definitions that can be referenced programmatically.
- The directory exposes importable Python modules rather than only opaque assets.

## Weaknesses / Risks

- No structural documentation risks were detected automatically; functional review is still required for behavior-level claims.

## Next Steps

- Keep this README synchronized with code and test changes through the managed documentation toolchain.

## References

- Parent README: [`doc_forge/src/doc_forge/scribe/README.md`](../README.md)
- Forge README: [`doc_forge/README.md`](../../../../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
