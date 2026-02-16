# Part 24: Frontier Upgrades

## Goal

Define optional but high-value research upgrades that remain operational, testable, and deployable in Termux/Linux.

## Upgrade 1: Dual-Channel SENSE

### Description

Separate exteroceptive and interoceptive sensing streams:

- `sense.external_percept`
- `sense.internal_percept`

### Why

Improves simulation-vs-grounded discrimination and world model calibration.

### Files

- `agent_forge/src/agent_forge/consciousness/modules/sense.py`
- `agent_forge/src/agent_forge/consciousness/modules/meta.py`

### Acceptance

- meta false-positive simulated classifications reduced in external-input-active trials.

## Upgrade 2: Precision-Weighted Predictive Coding

### Description

Introduce precision scalars modulating prediction-error impact based on affect/intero state.

### Why

Enables adaptive attentional regimes under stress/novelty.

### Files

- `agent_forge/src/agent_forge/consciousness/modules/world_model.py`
- `agent_forge/src/agent_forge/consciousness/modules/attention.py`
- `agent_forge/src/agent_forge/consciousness/modules/affect.py`

### Acceptance

- perturbing affect precision changes attention and policy in expected direction.

## Upgrade 3: Self-Other Discrimination Task

### Description

Create benchmark task where internal actions and external perturbations are confusable; system must infer ownership.

### Files

- `agent_forge/src/agent_forge/consciousness/bench/tasks.py`
- `agent_forge/src/agent_forge/consciousness/bench/scoring.py`

### Acceptance

- ownership classification accuracy > baseline chance and calibrated confidence under controlled task seeds.

## Upgrade 4: Lightweight Structural Causal Models

### Description

Add optional SCM proxy over module interaction graph for more explicit intervention diagnostics.

### Constraints

- keep optional and computationally bounded
- avoid heavy dependencies in core runtime

### Acceptance

- SCM diagnostics only active in benchmark mode, not default daemon cadence.

## Rollout Guidance

1. land each upgrade behind config flags
2. include explicit baseline comparisons
3. require no regression in existing parity smoke
