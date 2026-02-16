# Part 21: Phenomenology Probes (PPX)

## Goal

Operationalize phenomenology-like properties into measurable, falsifiable indices produced by the runtime itself.

## Status

- Completed in PR-H8.
- Integrated in default kernel runtime and surfaced in trial/benchmark snapshots.

## Deliverables

1. probe module computing windowed indices
2. `phenom.snapshot` events
3. optional `PHENOM` workspace broadcast
4. integration with benchmark scoring and report grounding

## Target Files

- `agent_forge/src/agent_forge/consciousness/modules/phenomenology_probe.py` (new)
- `agent_forge/src/agent_forge/consciousness/kernel.py`
- `agent_forge/src/agent_forge/consciousness/types.py`
- `agent_forge/src/agent_forge/consciousness/bench/scoring.py`
- `agent_forge/src/agent_forge/consciousness/bench/trials.py`
- `agent_forge/src/agent_forge/consciousness/trials.py`
- `agent_forge/src/agent_forge/consciousness/benchmarks.py`
- `agent_forge/tests/test_consciousness_phenomenology_probe.py`

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

## Runtime Surface

- Module name: `phenomenology_probe`
- Workspace kind: `PHENOM` (optional broadcast)
- Metric keys:
- `consciousness.phenom.unity_index`
- `consciousness.phenom.continuity_index`
- `consciousness.phenom.ownership_index`
- `consciousness.phenom.perspective_coherence_index`
- `consciousness.phenom.dream_likeness_index`
- Config keys:
- `phenom_scan_events`
- `phenom_window_seconds`
- `phenom_emit_interval_secs`
- `phenom_emit_delta_threshold`
- `phenom_unity_trace_threshold`
- `phenom_broadcast_enable`
- `phenom_broadcast_min_confidence`

## Validation

- PPX changes predictably under perturb recipes (`wm_lesion`, `sensory_deprivation`)
- PPX remains stable under no-op/control trials
- report and PPX coherence remain aligned

Executed in Termux:

```sh
PYTHONPATH=lib:agent_forge/src:memory_forge/src:knowledge_forge/src:eidos_mcp/src \
  eidosian_venv/bin/python -m pytest \
  agent_forge/tests/test_consciousness_phenomenology_probe.py \
  agent_forge/tests/test_consciousness_bench_trials.py -q
```

## Acceptance Criteria

1. PPX metrics are bounded and numerically stable.
2. at least three perturb recipes produce statistically separable PPX signatures.
3. PPX included in integrated benchmark and MCP status surfaces.
