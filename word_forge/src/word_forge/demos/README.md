# `word_forge/src/word_forge/demos`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:19:31Z`
- Path: `word_forge/src/word_forge/demos`

## What It Is

Managed directory documentation for `word_forge/src/word_forge/demos`, generated from tracked files and directory structure.

## Why It Exists

This directory exists to hold the `forge` surface for `word_forge` and keep that responsibility separate from adjacent forge concerns.

## How It Works

- Tracked files in scope: `17`
- Child directories: `0`
- Tests detected: `False`
- Python modules: `cli_demo, config_demo, conversation_demo, conversation_worker_demo, database_demo, database_worker_demo, emotion_demo, emotion_worker_demo, graph_demo, graph_worker_demo`

## Contents

- No managed child directories detected.

## Prominent Files

- [`__init__.py`](./__init__.py)
- [`cli_demo.py`](./cli_demo.py)
- [`config_demo.py`](./config_demo.py)
- [`conversation_demo.py`](./conversation_demo.py)
- [`conversation_worker_demo.py`](./conversation_worker_demo.py)
- [`database_demo.py`](./database_demo.py)
- [`database_worker_demo.py`](./database_worker_demo.py)
- [`emotion_demo.py`](./emotion_demo.py)

## Strengths

- The directory exposes importable Python modules rather than only opaque assets.

## Weaknesses / Risks

- No directly associated test coverage was detected under the tracked file set for this directory.
- The directory has many tracked files but no child-directory decomposition, which may make ownership blur over time.

## Next Steps

- Add focused tests or point this directory explicitly at its validating test surface.
- Keep this README synchronized with code and test changes through the managed documentation toolchain.

## References

- Parent README: [`word_forge/src/word_forge/README.md`](../README.md)
- Forge README: [`word_forge/README.md`](../../../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
