# `doc_forge/src/doc_forge/scribe`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:49:19Z`
- Path: `doc_forge/src/doc_forge/scribe`

## What It Is

Production document-processing pipeline: extract source context, generate docs, judge quality, persist state, and expose service APIs.

## Why It Exists

This is the operational heart of Doc Forge and the main integration point for living documentation automation.

## How It Works

- Tracked files in scope: `18`
- Child directories: `1`
- Tests detected: `True`
- Python modules: `config, directory_docs, extract, generate, judge, service, state`
- Detected API routes: `GET /; GET /api/docs/coverage; GET /api/docs/diff; GET /api/docs/readme; GET /api/docs/render; GET /api/status; GET /health`

## Contents

- [`tests`](./tests/README.md)

## Prominent Files

- [`README.md`](./README.md)
- [`__init__.py`](./__init__.py)
- [`config.py`](./config.py)
- [`directory_docs.py`](./directory_docs.py)
- [`extract.py`](./extract.py)
- [`generate.py`](./generate.py)
- [`judge.py`](./judge.py)
- [`service.py`](./service.py)

## API Surface

- `GET /`
- `GET /api/docs/coverage`
- `GET /api/docs/diff`
- `GET /api/docs/readme`
- `GET /api/docs/render`
- `GET /api/status`
- `GET /health`
- `POST /api/docs/upsert`
- `POST /api/docs/upsert-batch`

## Validating Tests

- [`doc_forge/src/doc_forge/scribe/tests/test_extract.py`](tests/test_extract.py)
- [`doc_forge/src/doc_forge/scribe/tests/test_generate.py`](tests/test_generate.py)
- [`doc_forge/src/doc_forge/scribe/tests/test_judge.py`](tests/test_judge.py)
- [`doc_forge/src/doc_forge/scribe/tests/test_service.py`](tests/test_service.py)
- [`doc_forge/src/doc_forge/scribe/tests/test_state.py`](tests/test_state.py)

## Strengths

- A directly associated test surface is present in or below this directory.
- Likely validating test files were matched from the surrounding forge test surface.
- The directory contains detected HTTP/API route definitions that can be referenced programmatically.
- The directory exposes importable Python modules rather than only opaque assets.
- Responsibility is split into child directories instead of one flat file heap.

## Weaknesses / Risks

- No structural documentation risks were detected automatically; functional review is still required for behavior-level claims.

## Next Steps

- Extend route coverage for documentation inventory and managed README generation.
- Benchmark generation quality and latency per directory class.

## References

- Parent README: [`doc_forge/src/doc_forge/README.md`](../README.md)
- Forge README: [`doc_forge/README.md`](../../../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
