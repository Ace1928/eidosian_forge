# `prompt_forge/src/prompt_forge`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:19:24Z`
- Path: `prompt_forge/src/prompt_forge`

## What It Is

Managed directory documentation for `prompt_forge/src/prompt_forge`, generated from tracked files and directory structure.

## Why It Exists

This directory exists to hold the `forge` surface for `prompt_forge` and keep that responsibility separate from adjacent forge concerns.

## How It Works

- Tracked files in scope: `3`
- Child directories: `1`
- Tests detected: `False`
- Python modules: `library`

## Contents

- [`cli`](./cli/README.md)

## Prominent Files

- [`__init__.py`](./__init__.py)
- [`library.py`](./library.py)

## Strengths

- The directory exposes importable Python modules rather than only opaque assets.
- Responsibility is split into child directories instead of one flat file heap.

## Weaknesses / Risks

- No directly associated test coverage was detected under the tracked file set for this directory.

## Next Steps

- Add focused tests or point this directory explicitly at its validating test surface.
- Keep this README synchronized with code and test changes through the managed documentation toolchain.

## References

- Parent README: [`prompt_forge/src/README.md`](../README.md)
- Forge README: [`prompt_forge/README.md`](../../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
