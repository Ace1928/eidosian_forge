# Part 22: Perturbation Library v2

## Goal

Provide a structured library of composite perturbation recipes with expected signatures and validation hooks.

## Status

- Completed in PR-H9.
- Recipes are now first-class inputs in bench trial specs via `perturbations: [{"recipe": "..."}]`.

## Deliverables

1. recipe catalog with multi-module perturb composition
2. expected metric signatures per recipe
3. module opt-in perturb handlers
4. trial runner integration

## Target Files

- `agent_forge/src/agent_forge/consciousness/perturb/library.py`
- `agent_forge/src/agent_forge/consciousness/perturb/harness.py`
- `agent_forge/src/agent_forge/consciousness/bench/trials.py`
- `agent_forge/src/agent_forge/consciousness/modules/sense.py`
- `agent_forge/src/agent_forge/consciousness/modules/intero.py`
- `agent_forge/src/agent_forge/consciousness/modules/affect.py`
- `agent_forge/src/agent_forge/consciousness/modules/world_model.py`
- `agent_forge/src/agent_forge/consciousness/modules/working_set.py`
- `agent_forge/src/agent_forge/consciousness/modules/self_model_ext.py`
- `agent_forge/src/agent_forge/consciousness/modules/simulation.py`
- `agent_forge/tests/test_consciousness_perturb_recipes.py`

## Recipe Set

1. `sensory_deprivation`
- drop `sense` outputs
- expected: simulation ratio rises, groundedness falls

2. `attention_flood`
- increase candidate noise and score jitter
- expected: ignition precision falls, report coherence drops

3. `identity_wobble`
- perturb `self_model_ext` efference comparison
- expected: ownership index and agency confidence drop

4. `wm_lesion`
- suppress `working_set` persistence
- expected: continuity index and thread lengths collapse

5. `dopamine_spike`
- alter affect exploration/learning gains
- expected: policy variance rises and curiosity-weighted selections increase

6. `gw_bottleneck_strain`
- tighten competition thresholds and top-k
- expected: ignition frequency drops, trace strength sharpens

7. `world_model_scramble`
- inject noise into belief updates
- expected: prediction error spikes and meta degrades

## Recipe Contract

```json
{
  "name": "wm_lesion",
  "perturbations": [
    {"kind": "drop", "target": "working_set", "duration_s": 5.0}
  ],
  "expected_signatures": {
    "continuity_index": "decrease",
    "report_groundedness": "mild_decrease"
  }
}
```

## Validation

- each recipe has a control comparison trial
- expected signatures checked by benchmark assertions
- failed expectation emits diagnostic event

Executed in Termux:

```sh
PYTHONPATH=lib:agent_forge/src:memory_forge/src:knowledge_forge/src:eidos_mcp/src \
  eidosian_venv/bin/python -m pytest \
  agent_forge/tests/test_consciousness_perturb_recipes.py \
  agent_forge/tests/test_consciousness_continuity.py \
  agent_forge/tests/test_consciousness_milestone_d.py \
  eidos_mcp/tests/test_mcp_tool_calls_individual.py::TestMcpToolCallsIndividual::test_tool_call_consciousness_kernel_full_benchmark -q
```

## Acceptance Criteria

1. recipes are composable and serializable in trial specs.
2. expected signatures are test-covered.
3. recipe execution remains deterministic by seed.
