# `glyph_forge/src/glyph_forge`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:19:22Z`
- Path: `glyph_forge/src/glyph_forge`

## What It Is

Managed directory documentation for `glyph_forge/src/glyph_forge`, generated from tracked files and directory structure.

## Why It Exists

This directory exists to hold the `forge` surface for `glyph_forge` and keep that responsibility separate from adjacent forge concerns.

## How It Works

- Tracked files in scope: `50`
- Child directories: `10`
- Tests detected: `False`
- Python modules: `__main__, eidos_profile`

## Contents

- [`api`](./api/README.md)
- [`cli`](./cli/README.md)
- [`config`](./config/README.md)
- [`core`](./core/README.md)
- [`renderers`](./renderers/README.md)
- [`services`](./services/README.md)
- [`streaming`](./streaming/README.md)
- [`transformers`](./transformers/README.md)
- [`ui`](./ui/README.md)
- [`utils`](./utils/README.md)

## Prominent Files

- [`__init__.py`](./__init__.py)
- [`__main__.py`](./__main__.py)
- [`eidos_profile.py`](./eidos_profile.py)
- [`py.typed`](./py.typed)

## Strengths

- The directory exposes importable Python modules rather than only opaque assets.
- Responsibility is split into child directories instead of one flat file heap.

## Weaknesses / Risks

- No directly associated test coverage was detected under the tracked file set for this directory.

## Next Steps

- Add focused tests or point this directory explicitly at its validating test surface.
- Keep this README synchronized with code and test changes through the managed documentation toolchain.

## References

- Parent README: [`glyph_forge/src/README.md`](../README.md)
- Forge README: [`glyph_forge/README.md`](../../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
