# `computer_control_forge/src/computer_control_forge`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:19:21Z`
- Path: `computer_control_forge/src/computer_control_forge`

## What It Is

Managed directory documentation for `computer_control_forge/src/computer_control_forge`, generated from tracked files and directory structure.

## Why It Exists

This directory exists to hold the `forge` surface for `computer_control_forge` and keep that responsibility separate from adjacent forge concerns.

## How It Works

- Tracked files in scope: `32`
- Child directories: `4`
- Tests detected: `False`
- Python modules: `absolute_mouse, actions, advanced_mouse, agent_controller, agent_loop, calibrated_mouse, control, cursor_position, feedback_system, human_mouse`

## Contents

- [`automation`](./automation/README.md)
- [`cli`](./cli/README.md)
- [`daemon`](./daemon/README.md)
- [`perception`](./perception/README.md)

## Prominent Files

- [`__init__.py`](./__init__.py)
- [`absolute_mouse.py`](./absolute_mouse.py)
- [`actions.py`](./actions.py)
- [`advanced_mouse.py`](./advanced_mouse.py)
- [`agent_controller.py`](./agent_controller.py)
- [`agent_loop.py`](./agent_loop.py)
- [`calibrated_mouse.py`](./calibrated_mouse.py)
- [`control.py`](./control.py)

## Strengths

- The directory exposes importable Python modules rather than only opaque assets.
- Responsibility is split into child directories instead of one flat file heap.

## Weaknesses / Risks

- No directly associated test coverage was detected under the tracked file set for this directory.

## Next Steps

- Add focused tests or point this directory explicitly at its validating test surface.
- Keep this README synchronized with code and test changes through the managed documentation toolchain.

## References

- Parent README: [`computer_control_forge/src/README.md`](../README.md)
- Forge README: [`computer_control_forge/README.md`](../../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
