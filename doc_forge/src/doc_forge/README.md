# `doc_forge/src/doc_forge`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:19:21Z`
- Path: `doc_forge/src/doc_forge`

## What It Is

Primary Doc Forge Python package for repository-scale documentation extraction, generation, judging, and service orchestration.

## Why It Exists

It centralizes documentation production logic so documentation can be treated as a first-class maintained system instead of ad hoc markdown.

## How It Works

- Tracked files in scope: `39`
- Child directories: `6`
- Tests detected: `True`
- Python modules: `__main__, autoapi_fixer, doc_forge, doc_manifest_manager, doc_migration, doc_toc_analyzer, doc_validator, fix_cross_refs, fix_docstrings, fix_duplicate_objects`
- Detected API routes: `GET /; GET /api/docs/coverage; GET /api/docs/diff; GET /api/docs/readme; GET /api/docs/render; GET /api/status; GET /health; POST /api/docs/upsert`

## Contents

- [`api`](./api/README.md)
- [`config`](./config/README.md)
- [`core`](./core/README.md)
- [`scribe`](./scribe/README.md)
- [`services`](./services/README.md)
- [`utils`](./utils/README.md)

## Prominent Files

- [`__init__.py`](./__init__.py)
- [`__main__.py`](./__main__.py)
- [`autoapi_fixer.py`](./autoapi_fixer.py)
- [`doc_forge.py`](./doc_forge.py)
- [`doc_manifest_manager.py`](./doc_manifest_manager.py)
- [`doc_migration.py`](./doc_migration.py)
- [`doc_toc_analyzer.py`](./doc_toc_analyzer.py)
- [`doc_validator.py`](./doc_validator.py)

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

## Strengths

- A directly associated test surface is present in or below this directory.
- The directory contains detected HTTP/API route definitions that can be referenced programmatically.
- The directory exposes importable Python modules rather than only opaque assets.
- Responsibility is split into child directories instead of one flat file heap.

## Weaknesses / Risks

- No structural documentation risks were detected automatically; functional review is still required for behavior-level claims.

## Next Steps

- Expand the documentation contract deeper across package subdirectories.
- Keep API and service claims synchronized with test coverage.

## References

- Parent README: [`doc_forge/src/README.md`](../README.md)
- Forge README: [`doc_forge/README.md`](../../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
