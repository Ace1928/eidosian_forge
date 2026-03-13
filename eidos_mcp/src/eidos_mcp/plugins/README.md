# `eidos_mcp/src/eidos_mcp/plugins`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:19:21Z`
- Path: `eidos_mcp/src/eidos_mcp/plugins`

## What It Is

Managed directory documentation for `eidos_mcp/src/eidos_mcp/plugins`, generated from tracked files and directory structure.

## Why It Exists

This directory exists to hold the `directory` surface for `eidos_mcp` and keep that responsibility separate from adjacent forge concerns.

## How It Works

- Tracked files in scope: `7`
- Child directories: `4`
- Tests detected: `False`
- Python modules: `core`

## Contents

- [`computer_control`](./computer_control/README.md)
- [`self_exploration`](./self_exploration/README.md)
- [`task_automation`](./task_automation/README.md)
- [`web_tools`](./web_tools/README.md)

## Prominent Files

- [`__init__.py`](./__init__.py)
- [`core.py`](./core.py)

## Strengths

- The directory exposes importable Python modules rather than only opaque assets.
- Responsibility is split into child directories instead of one flat file heap.

## Weaknesses / Risks

- No directly associated test coverage was detected under the tracked file set for this directory.

## Next Steps

- Add focused tests or point this directory explicitly at its validating test surface.
- Keep this README synchronized with code and test changes through the managed documentation toolchain.

## References

- Parent README: [`eidos_mcp/src/eidos_mcp/README.md`](../README.md)
- Forge README: [`eidos_mcp/README.md`](../../../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
