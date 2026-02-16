# Part 32: Integrated Red-Team Gates in Full Benchmark

## Goal

Promote adversarial robustness from a standalone campaign into a first-class signal inside the integrated benchmark score and gate set.

## Implementation

- Integrated benchmark runtime now accepts red-team controls:
  - `run_red_team`
  - `red_team_quick`
  - `red_team_max_scenarios`
  - `red_team_seed`
- Full benchmark report now includes:
  - `red_team` section with scenario outcomes and robustness metrics
  - `scores.red_team_score`
  - gate predicates: `red_team_available`, `red_team_pass_min`, `red_team_robustness_min`
- Score aggregation upgraded to active-component weighted normalization.
  - Disabled channels no longer implicitly penalize integrated score.

## Surface Integration

- CLI `eidctl consciousness full-benchmark`:
  - `--skip-red-team`
  - `--red-team-quick`
  - `--red-team-max-scenarios`
  - `--red-team-seed`
- MCP `consciousness_kernel_full_benchmark` now accepts matching red-team parameters.

## Validation

- Added integrated benchmark regression with quick red-team mode.
- Updated CLI full-benchmark test path to explicitly skip red-team for deterministic fast test cycles.
- Updated MCP individual full-benchmark call arguments to pin `run_red_team=false` in standard tool-matrix tests.

## Rationale

This turns adversarial stress from a separate report into an optimization-relevant runtime constraint, making "full benchmark" genuinely represent end-to-end safety, coherence, and robustness.
