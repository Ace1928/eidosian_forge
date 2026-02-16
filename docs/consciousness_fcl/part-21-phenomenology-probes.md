# Part 21: Phenomenology Probes (PPX)

## Goal

Operationalize phenomenology-like properties into measurable, falsifiable indices produced by the runtime itself.

## Deliverables

1. probe module computing windowed indices
2. `phenom.snapshot` events
3. optional `PHENOM` workspace broadcast
4. integration with benchmark scoring and report grounding

## Target Files

- `agent_forge/src/agent_forge/consciousness/modules/phenomenology_probe.py` (new)
- `agent_forge/src/agent_forge/consciousness/metrics/self_stability.py`
- `agent_forge/src/agent_forge/consciousness/bench/scoring.py`

## Probe Indices

1. `unity_index`
- fraction of windows with ignition trace strength above threshold

2. `continuity_index`
- working set survival time
- re-entry rates
- thread length continuity

3. `ownership_index`
- agency confidence mean and perturbation sensitivity

4. `perspective_coherence_index`
- report claims with resolvable supporting links

5. `dream_likeness_index`
- simulated percept dominance + mode agreement + grounding mismatch

## Snapshot Schema

```json
{
  "window": {"seconds": 10},
  "unity_index": 0.72,
  "continuity_index": 0.64,
  "ownership_index": 0.58,
  "perspective_coherence_index": 0.69,
  "dream_likeness_index": 0.11,
  "evidence": {"winner_count": 14, "report_count": 8}
}
```

## Integration Plan

- emit `phenom.snapshot` every configured probe window
- include latest PPX indices in benchmark report payloads
- allow ablation tests to assert expected PPX shifts

## Validation

- PPX changes predictably under perturb recipes (`wm_lesion`, `sensory_deprivation`)
- PPX remains stable under no-op/control trials
- report and PPX coherence remain aligned

## Acceptance Criteria

1. PPX metrics are bounded and numerically stable.
2. at least three perturb recipes produce statistically separable PPX signatures.
3. PPX included in integrated benchmark and MCP status surfaces.
