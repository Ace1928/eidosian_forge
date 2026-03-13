# `lib/eidosian_runtime`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:19:22Z`
- Path: `lib/eidosian_runtime`

## What It Is

Shared runtime coordination helpers for scheduler, capability registry, and service-state governance.

## Why It Exists

It keeps platform/runtime checks out of leaf modules and gives Atlas, boot scripts, and scheduler one shared contract.

## How It Works

- Tracked files in scope: `3`
- Child directories: `0`
- Tests detected: `False`
- Python modules: `capabilities, coordinator`

## Contents

- No managed child directories detected.

## Prominent Files

- [`__init__.py`](./__init__.py)
- [`capabilities.py`](./capabilities.py)
- [`coordinator.py`](./coordinator.py)

## Strengths

- The directory exposes importable Python modules rather than only opaque assets.

## Weaknesses / Risks

- No directly associated test coverage was detected under the tracked file set for this directory.

## Next Steps

- Continue moving platform checks out of leaf modules into capability-driven paths.
- Expose longer-lived runtime history and policy state through the same contract.

## References

- Parent README: [`lib/README.md`](../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
