# Part 28: Parameter Store and Overlay Control Plane

## Goal

Formalize machine-tunable parameters with explicit ranges, safety classes, and typed coercion so self-upgrades are deterministic and auditable.

## Schema

`ParamSpec` fields:

- `key`
- `kind` (`float|int|bool|choice`)
- `default`
- `min_value`, `max_value`
- `choices`
- `safety`
- `description`

## Initial Tunable Surface

Competition:

- `competition_top_k`
- `competition_min_score`
- `competition_trace_strength_threshold`
- `competition_reaction_window_secs`
- `competition_cooldown_secs`

World model:

- `world_belief_alpha`
- `world_error_broadcast_threshold`
- `world_error_derivative_threshold`
- `world_prediction_window`

Meta:

- `meta_observation_window`
- `meta_emit_delta_threshold`

Cadence:

- `module_tick_periods.world_model`
- `module_tick_periods.meta`
- `module_tick_periods.report`
- `module_tick_periods.phenomenology_probe`

## Overlay Resolution

1. Validate/sanitize overlay against parameter specs.
2. Persist sanitized overlay in state store metadata.
3. Resolve runtime config by precedence:
   `runtime_overrides > tuned_overlay > base_config`.

## Persistence

- `tuned_overlay`
- `tuned_overlay_version`
- `tuned_overlay_history` (bounded, newest retained)

## Defensive Behavior

- Unknown keys are ignored and reported.
- Out-of-range values are clamped to spec ranges.
- Invalid types are coerced to defaults.

## Acceptance Criteria

1. Overlay round-trips through state store without schema drift.
2. Invalid keys never crash runtime; they emit diagnostics.
3. Nested keys (`module_tick_periods.*`) resolve deterministically.

