# Part 23: Ablation Matrix, Golden Ranges, and Regression Gates

## Goal

Prove each module contributes meaningful dynamics by running repeatable ablations and enforcing expected metric deltas.

## Deliverables

1. ablation matrix runner
2. golden range store for key metrics
3. CI gates for non-regression and contribution sanity

## Target Files

- `agent_forge/src/agent_forge/consciousness/bench/ablations.py`
- `agent_forge/src/agent_forge/consciousness/bench/golden.py`
- `agent_forge/tests/test_consciousness_ablations.py` (new)
- `scripts/linux_parity_smoke.sh` (matrix extension)

## Ablation Matrix

Run each `TrialSpec` under:

- full stack
- minus `workspace_competition`
- minus `working_set`
- minus `world_model`
- minus `self_model_ext`
- minus `meta`
- minus `intero` + `affect`

## Golden Ranges

Store expected ranges per metric and trial family:

```json
{
  "trial_family": "baseline_gnw",
  "metric_ranges": {
    "trace_strength_median": [0.35, 0.85],
    "report_groundedness_median": [0.45, 0.95],
    "ownership_index_median": [0.35, 0.9]
  }
}
```

## Expected Delta Assertions

Examples:

- no competition -> ignition trace median < threshold
- no working set -> continuity index drops >= 50%
- no self-model -> ownership index collapses
- no world model -> simulation/meta alignment degrades

## CI Strategy

- smoke ablation set for every push
- full ablation grid on schedule/nightly
- output trends persisted for review

## Acceptance Criteria

1. every major module has at least one measurable contribution assertion.
2. golden range updates are explicit and reviewed, not implicit drift.
3. benchmark pipeline fails on unexpected metric regressions.
