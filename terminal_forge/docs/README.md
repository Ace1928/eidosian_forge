# `terminal_forge/docs`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:19:25Z`
- Path: `terminal_forge/docs`

## What It Is

Managed directory documentation for `terminal_forge/docs`, generated from tracked files and directory structure.

## Why It Exists

This directory exists to hold the `docs` surface for `terminal_forge` and keep that responsibility separate from adjacent forge concerns.

## How It Works

- Tracked files in scope: `162`
- Child directories: `10`
- Tests detected: `False`
- Python modules: `conf`

## Contents

- [`_static`](./_static/README.md)
- [`_templates`](./_templates/README.md)
- [`ai`](./ai/README.md)
- [`assets`](./assets/README.md)
- [`auto`](./auto/README.md)
- [`examples`](./examples/README.md)
- [`manual`](./manual/README.md)
- [`source`](./source/README.md)
- [`tools`](./tools/README.md)
- [`versions`](./versions/README.md)

## Prominent Files

- [`.readthedocs.yaml`](./.readthedocs.yaml)
- [`api_reference.md`](./api_reference.md)
- [`conf.py`](./conf.py)
- [`examples.md`](./examples.md)
- [`getting_started.md`](./getting_started.md)
- [`index.md`](./index.md)

## Strengths

- The directory exposes importable Python modules rather than only opaque assets.
- Responsibility is split into child directories instead of one flat file heap.

## Weaknesses / Risks

- No directly associated test coverage was detected under the tracked file set for this directory.

## Next Steps

- Add focused tests or point this directory explicitly at its validating test surface.
- Keep this README synchronized with code and test changes through the managed documentation toolchain.

## References

- Parent README: [`terminal_forge/README.md`](../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
