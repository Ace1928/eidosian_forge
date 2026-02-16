# Part 29: Autotune Runtime Module

## Goal

Run controlled, low-frequency self-optimization cycles directly inside the consciousness runtime without breaking safety or reproducibility.

## Module

- File: `agent_forge/src/agent_forge/consciousness/modules/autotune.py`
- Name: `autotune`
- Cadence: `module_tick_periods.autotune` (default `60` beats) and `autotune_interval_beats` (default `120` beats)

## Loop

1. Safety check:
   - no active perturbations
   - meta mode is not degraded
   - recent module error count below threshold
2. Baseline trial bootstrapping if missing baseline score.
3. Candidate overlay proposal using bandit optimizer.
4. Candidate micro-trial execution via bench runner.
5. Guardrail evaluation + score improvement test.
6. Commit tuned overlay or rollback and emit corresponding event.

## Config Controls

- `autotune_enabled`
- `autotune_interval_beats`
- `autotune_min_improvement`
- `autotune_task`
- `autotune_trial_*`
- `autotune_guardrail_*`

## Trial Requirements

Micro-trials are executed with:

- deterministic seed offsets
- module disable list including `autotune` (prevents recursion)
- optional artifact persistence for forensic analysis

## Metrics

- `consciousness.autotune.best_score`
- `consciousness.autotune.last_score`
- `consciousness.autotune.acceptance_ratio`

## Acceptance Criteria

1. Successful proposals emit `tune.commit` and increment overlay version.
2. Unsafe or non-improving proposals emit `tune.rollback`.
3. Guardrail violations never commit overlays.
4. Runtime remains responsive (autotune is low-frequency and bounded).

