# `doc_forge/src`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:19:21Z`
- Path: `doc_forge/src`

## What It Is

Managed directory documentation for `doc_forge/src`, generated from tracked files and directory structure.

## Why It Exists

This directory exists to hold the `src` surface for `doc_forge` and keep that responsibility separate from adjacent forge concerns.

## How It Works

- Tracked files in scope: `63`
- Child directories: `4`
- Tests detected: `True`
- Detected API routes: `GET /; GET /api/docs/coverage; GET /api/docs/diff; GET /api/docs/readme; GET /api/docs/render; GET /api/status; GET /health; POST /api/docs/upsert`

## Contents

- [`doc_forge`](./doc_forge/README.md)
- [`examples`](./examples/README.md)
- [`templates`](./templates/README.md)
- [`tests`](./tests/README.md)

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
- Responsibility is split into child directories instead of one flat file heap.

## Weaknesses / Risks

- No structural documentation risks were detected automatically; functional review is still required for behavior-level claims.

## Next Steps

- Keep this README synchronized with code and test changes through the managed documentation toolchain.

## References

- Parent README: [`doc_forge/README.md`](../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
