# Part 04: Core Modules

## Goal

Implement primary modules in ascending dependency order.

## Order

1. `attention`
2. `workspace_competition`
3. `policy`
4. `self_model_ext`
5. `world_model`
6. `meta`
7. `report`

## Implementation Rules

- Emit structured events for all outputs.
- Broadcast only selected payloads into workspace.
- Keep deterministic defaults and seeded RNG for test reproducibility.

## Acceptance Criteria

- Competition produces winners and `gw.ignite` markers.
- Policy emits efference and self module derives agency confidence.
- Report module references evidence IDs from recent events.
