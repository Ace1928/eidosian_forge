# Part 20: Simulation Stream - Internal Generative Percepts

## Goal

Introduce a controlled simulated percept stream ("dreaming") driven by world model rollouts and validated by meta-state recognition.

## Deliverables

1. simulation module with mode-gated rollouts
2. `sense.simulated_percept` event stream
3. attention integration with confidence penalties
4. report/meta awareness of simulated content

## Target Files

- `agent_forge/src/agent_forge/consciousness/modules/simulation.py` (new)
- `agent_forge/src/agent_forge/consciousness/modules/meta.py` (integration)
- `agent_forge/src/agent_forge/consciousness/modules/report.py` (labeling integration)
- `agent_forge/src/agent_forge/consciousness/kernel.py` (module order/cadence)

## Trigger Conditions

Simulation activates when one or more conditions hold:

- exteroceptive inputs are sparse for configured interval
- meta mode indicates `simulated`
- explicit simulation perturbation recipe enabled

Config keys:

- `simulation_quiet_window_beats`
- `simulation_max_percepts_per_tick`
- `simulation_confidence_penalty`

## Event Schema

` sense.simulated_percept` payload:

```json
{
  "simulated": true,
  "origin": "world_model.rollout",
  "rollout_step": 2,
  "belief_snapshot_id": "...",
  "predicted_kind": "PERCEPT",
  "confidence": 0.44,
  "salience": 0.31,
  "links": {"corr_id": "...", "parent_id": "..."}
}
```

## Integration Rules

- simulation candidates can compete in workspace competition
- simulated winners must be explicitly tagged in winner payload content
- report module includes simulated fraction and disconfirmation note

## Validation

- sensory deprivation perturbation increases simulated percept ratio
- meta mode classification shifts to `simulated` with confidence increase
- report correctly labels simulated evidence paths

## Acceptance Criteria

1. simulation stream is deterministic under fixed seed and inputs.
2. simulation does not override external percepts when exteroception is available.
3. report/meta show consistent simulated-state annotations.
