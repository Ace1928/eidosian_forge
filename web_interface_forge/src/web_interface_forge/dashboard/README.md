# `web_interface_forge/src/web_interface_forge/dashboard`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:19:31Z`
- Path: `web_interface_forge/src/web_interface_forge/dashboard`

## What It Is

Atlas dashboard backend and templates for runtime control, graph exploration, and operator-facing forge observability.

## Why It Exists

It gives the forge an operator plane instead of requiring shell-only control and log inspection.

## How It Works

- Tracked files in scope: `7`
- Child directories: `2`
- Tests detected: `False`
- Python modules: `main`
- Detected API routes: `GET /; GET /api/capabilities; GET /api/doc/status; GET /api/runtime; GET /api/runtime/local-agent; GET /api/scheduler; GET /api/services; GET /api/system`

## Contents

- [`static`](./static/README.md)
- [`templates`](./templates/README.md)

## Prominent Files

- [`__init__.py`](./__init__.py)
- [`main.py`](./main.py)

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

- Expose documentation coverage and diff workflows directly in Atlas.
- Continue converging service, scheduler, and documentation control in one surface.

## References

- Parent README: [`web_interface_forge/src/web_interface_forge/README.md`](../README.md)
- Forge README: [`web_interface_forge/README.md`](../../../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
