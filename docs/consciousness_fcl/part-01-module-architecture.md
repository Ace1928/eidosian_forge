# Part 01: Module Architecture

## Goal

Add additive package architecture in `agent_forge` with deterministic interfaces.

## Package Layout

`agent_forge/src/agent_forge/consciousness/`
- `types.py`
- `kernel.py`
- `modules/`
- `metrics/`
- `perturb/`

## Required Interfaces

- `TickContext`
- `Module` protocol with `name` and `tick(ctx)`
- event helpers for append/broadcast/metric emission

## Implementation Notes

- Keep strict stdlib for core module internals unless existing dependencies are already in project.
- Preserve compatibility with daemon loop and existing state/event contracts.

## Acceptance Criteria

- Kernel can execute module ticks with deterministic ordering.
- Module failures degrade gracefully and emit error events.
