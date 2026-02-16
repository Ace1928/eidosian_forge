# Part 18: Metrics Suite v2 - Connectivity, Directionality, Stability

Status: core metrics v2 package delivered in Stage H PR-H4 (`rci_v2`, `entropy`, `connectivity`, `directionality`, `self_stability`).

## Goal

Add richer, falsifiable measurement primitives beyond basic RCI/coherence.

## Deliverables

1. `RCI_v2` (entropy + temporal structure + compression)
2. effective connectivity graph metrics
3. directional asymmetry proxy
4. self-binding stability metrics

## Target Files

- `agent_forge/src/agent_forge/consciousness/metrics/rci.py` (extend)
- `agent_forge/src/agent_forge/consciousness/metrics/entropy.py` (new)
- `agent_forge/src/agent_forge/consciousness/metrics/connectivity.py` (new)
- `agent_forge/src/agent_forge/consciousness/metrics/self_stability.py` (new)

## RCI_v2 Components

- event-type entropy
- source entropy
- lag-1 transition structure
- burstiness / Fano factor proxy
- compression ratio

Emit:

- `metrics.rci_v2`
- subcomponents for diagnostics

## Effective Connectivity Graph

Graph definition:

- node: module
- directed edge A->B when B emits linked reaction after A winner/broadcast within `delta_t`

Metrics:

- edge density
- strongly connected component stats
- hubness distribution
- centrality of workspace competition node

Emit:

- `metrics.connectivity.*`
- optional summarized graph snapshot events

## Directionality Proxy

Compute lagged mutual information asymmetry:

`MI(A_t, B_t+1) - MI(B_t, A_t+1)`

Used as directionality signal, not a formal causality proof.

Emit:

- `metrics.directionality.<module_a>.<module_b>`

## Self-Binding Stability

Track stability over rolling windows:

- agency confidence variance
- boundary stability variance
- perturbation sensitivity deltas

Emit:

- `metrics.self_binding_stability`

## Validation

- unit tests for metric boundedness and expected ranges
- synthetic known patterns (random/noisy/structured) produce expected ordering
- ablation deltas follow predicted signatures

## Acceptance Criteria

1. metric outputs remain stable under replay for deterministic inputs.
2. connectivity metrics discriminate full-stack vs ablated-stack behavior.
3. benchmark reports include v2 metrics with trend comparators.
