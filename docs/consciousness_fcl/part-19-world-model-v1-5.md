# Part 19: World Model v1.5 - Belief State and Predictive Coding

Status: core PR-H6 delivery complete (`features.py` + upgraded `world_model.py` + rollout and belief-state emissions).

## Goal

Upgrade world modeling from event-type transitions to feature-based predictive coding with persistent belief dynamics and rollout capability.

## Deliverables

1. feature extraction pipeline `phi(event)`
2. persistent belief state `b_t`
3. prediction error decomposition with top contributing features
4. rollout API for simulation mode

## Target Files

- `agent_forge/src/agent_forge/consciousness/modules/world_model.py`
- `agent_forge/src/agent_forge/consciousness/features.py` (new)

## Feature Representation

`phi(event)` includes:

- event type token
- source module token
- payload `kind`
- salience/confidence numeric values
- drive fields from intero
- working set load
- ignition trace strength
- temporal delta features

## Belief Update

EMA-style state:

```text
b[k] = (1-alpha)*b[k] + alpha*phi[k]
```

Config:

- `world_model_alpha`
- `world_model_precision_min`
- `world_model_precision_max`

## Prediction and Error

- predict next features from current belief
- compute error by feature group
- emit top-k contributing mismatches

Events:

- `world.belief_state`
- `world.prediction`
- `world.prediction_error`
- `world.surprise`

## Surprise Gating

Broadcast `PRED_ERR` only when:

- absolute surprise > threshold
- or surprise derivative > threshold

This avoids flooding the workspace with low-value model chatter.

## Rollout API

```python
def rollout(self, steps: int, policy_hint: dict | None = None) -> list[dict[str, Any]]:
    ...
```

Outputs predicted feature packets tagged as simulated candidates for downstream modules.

## Validation

- prediction error decreases over structured training windows
- feature-specific errors spike under targeted perturbations
- rollouts produce coherent simulated percept seeds

## Acceptance Criteria

1. world model emits belief/prediction events every configured cadence.
2. top feature error decomposition is present in surprise events.
3. simulation module can consume rollouts without extra adapters.
