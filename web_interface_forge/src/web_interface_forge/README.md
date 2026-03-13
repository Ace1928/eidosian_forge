# `web_interface_forge/src/web_interface_forge`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:19:31Z`
- Path: `web_interface_forge/src/web_interface_forge`

## What It Is

Managed directory documentation for `web_interface_forge/src/web_interface_forge`, generated from tracked files and directory structure.

## Why It Exists

This directory exists to hold the `forge` surface for `web_interface_forge` and keep that responsibility separate from adjacent forge concerns.

## How It Works

- Tracked files in scope: `12`
- Child directories: `2`
- Tests detected: `False`
- Python modules: `eidos_client, eidos_server`
- Detected API routes: `GET /; GET /api/capabilities; GET /api/doc/status; GET /api/runtime; GET /api/runtime/local-agent; GET /api/scheduler; GET /api/services; GET /api/system`

## Contents

- [`cli`](./cli/README.md)
- [`dashboard`](./dashboard/README.md)

## Prominent Files

- [`README.md`](./README.md)
- [`__init__.py`](./__init__.py)
- [`eidos_client.py`](./eidos_client.py)
- [`eidos_server.py`](./eidos_server.py)

## API Surface

- `GET /`
- `GET /api/capabilities`
- `GET /api/doc/status`
- `GET /api/runtime`
- `GET /api/runtime/local-agent`
- `GET /api/scheduler`
- `GET /api/services`
- `GET /api/system`
- `GET /browse/{domain}/`
- `GET /browse/{domain}/{path:path}`
- `GET /browse/{path:path}`
- `GET /health`
- `POST /api/scheduler/{action}`
- `POST /api/services/{action}`

## Strengths

- The directory contains detected HTTP/API route definitions that can be referenced programmatically.
- The directory exposes importable Python modules rather than only opaque assets.
- Responsibility is split into child directories instead of one flat file heap.

## Weaknesses / Risks

- No directly associated test coverage was detected under the tracked file set for this directory.

## Next Steps

- Add focused tests or point this directory explicitly at its validating test surface.
- Keep this README synchronized with code and test changes through the managed documentation toolchain.

## References

- Parent README: [`web_interface_forge/src/README.md`](../README.md)
- Forge README: [`web_interface_forge/README.md`](../../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
