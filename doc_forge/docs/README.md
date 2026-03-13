# `doc_forge/docs`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:19:21Z`
- Path: `doc_forge/docs`

## What It Is

Managed directory documentation for `doc_forge/docs`, generated from tracked files and directory structure.

## Why It Exists

This directory exists to hold the `docs` surface for `doc_forge` and keep that responsibility separate from adjacent forge concerns.

## How It Works

- Tracked files in scope: `46`
- Child directories: `5`
- Tests detected: `False`
- Python modules: `conf`

## Contents

- [`_static`](./_static/README.md)
- [`_templates`](./_templates/README.md)
- [`assets`](./assets/README.md)
- [`auto_docs`](./auto_docs/README.md)
- [`user_docs`](./user_docs/README.md)

## Prominent Files

- [`.gitkeep`](./.gitkeep)
- [`.readthedocs.yaml`](./.readthedocs.yaml)
- [`conf.py`](./conf.py)
- [`docs_manifest.json`](./docs_manifest.json)
- [`index.md`](./index.md)
- [`requirements.txt`](./requirements.txt)

## Strengths

- The directory exposes importable Python modules rather than only opaque assets.
- Responsibility is split into child directories instead of one flat file heap.

## Weaknesses / Risks

- No directly associated test coverage was detected under the tracked file set for this directory.

## Next Steps

- Add focused tests or point this directory explicitly at its validating test surface.
- Keep this README synchronized with code and test changes through the managed documentation toolchain.

## References

- Parent README: [`doc_forge/README.md`](../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
