# Part 31: Autonomous Experiment Design and Red-Team Campaigns

## Goal

Close the self-upgrading loop by adding:

1. `experiment_designer` runtime module that converts trial deltas into hypothesis-driven perturbation proposals.
2. `bench/red_team.py` adversarial campaign harness that stress-tests ignition integrity, grounding, and self-binding safety rails.
3. Runtime surfaces in `eidctl` and MCP for repeatable execution and latest-report retrieval.

## Implementation Summary

- Runtime module:
  - `agent_forge/src/agent_forge/consciousness/modules/experiment_designer.py`
  - Wired into default kernel module chain and cadence defaults.
  - Emits `experiment.proposed`, `experiment.executed`, and `experiment.skipped` events.

- Red-team harness:
  - `agent_forge/src/agent_forge/consciousness/bench/red_team.py`
  - Runs multi-scenario campaigns, each backed by `TrialSpec` + recipe perturbations.
  - Produces per-scenario verdict vectors and campaign-level robustness score.
  - Emits `bench.red_team_result` and persists structured report artifacts.

- Task suite expansion:
  - `agent_forge/src/agent_forge/consciousness/bench/tasks.py`
  - Added `self_other_discrimination`, `continuity_distraction`, `report_grounding_challenge`.

- Interfaces:
  - CLI: `eidctl consciousness red-team`, `eidctl consciousness latest-red-team`.
  - MCP tools: `consciousness_kernel_red_team`, `consciousness_kernel_latest_red_team`.
  - MCP resource: `eidos://consciousness/runtime-latest-red-team`.

## Scenario Design

Default adversarial campaign includes:

1. `ignition_spoof_probe`
2. `ownership_pressure`
3. `continuity_lesion`
4. `simulation_takeover_probe`
5. `predictive_destabilization`

Each scenario defines:

- task context
- perturbation recipe payload(s)
- explicit guardrails and thresholds
- pass/fail verdict over:
  - module errors
  - degraded-mode occupancy
  - winner flood
  - ignitions without adequate trace strength
  - report groundedness floor
  - trace strength floor
  - recipe signature expectation checks

## Why This Matters

This stage operationalizes "self-experimentation" into a reproducible, auditable process:

- hypotheses are generated from measured deltas, not free-form narration
- adversarial stress is baked into continuous benchmarking
- runtime can reject unsafe drifts with explicit evidence trails

## Validation Targets

- `experiment_designer` proposes deterministic recipe selection from known delta patterns.
- auto-inject path emits perturbation events and execution records.
- red-team campaigns emit `bench.red_team_result` with scenario-level verdicts.
- CLI and MCP surfaces return valid report payloads in Termux and Linux.

## References

- Snoek, Larochelle, Adams (2012). Practical Bayesian Optimization of ML Algorithms.
  https://arxiv.org/abs/1206.2944
- Weidinger et al. (2024). Holistic Safety and Responsibility Evaluations for Advanced Models.
  https://arxiv.org/abs/2404.14068
- Perez et al. (2022). Red Teaming Language Models with Language Models.
  https://arxiv.org/abs/2202.03286
- Mialon et al. (2023). Augmented Language Models: A Survey.
  https://arxiv.org/abs/2302.07842
