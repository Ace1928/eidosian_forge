# `metadata_forge/tests`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:19:24Z`
- Path: `metadata_forge/tests`

## What It Is

Managed directory documentation for `metadata_forge/tests`, generated from tracked files and directory structure.

## Why It Exists

This directory exists to hold the `tests` surface for `metadata_forge` and keep that responsibility separate from adjacent forge concerns.

## How It Works

- Tracked files in scope: `9`
- Child directories: `8`
- Tests detected: `True`
- Python modules: `test_schema`

## Contents

- [`e2e`](./e2e/README.md)
- [`fixtures`](./fixtures/README.md)
- [`integration`](./integration/README.md)
- [`mocks`](./mocks/README.md)
- [`performance`](./performance/README.md)
- [`security`](./security/README.md)
- [`unit`](./unit/README.md)
- [`utils`](./utils/README.md)

## Prominent Files

- [`test_schema.py`](./test_schema.py)

## Strengths

- A directly associated test surface is present in or below this directory.
- The directory exposes importable Python modules rather than only opaque assets.
- Responsibility is split into child directories instead of one flat file heap.

## Weaknesses / Risks

- No structural documentation risks were detected automatically; functional review is still required for behavior-level claims.

## Next Steps

- Keep this README synchronized with code and test changes through the managed documentation toolchain.

## References

- Parent README: [`metadata_forge/README.md`](../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
