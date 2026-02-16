# Part 26: Adversarial Self Red-Team Suite

## Goal

Make the system adversarial to itself so improvements survive failure-inducing tests, not just cooperative scenarios.

## Deliverables

1. red-team perturbation campaign runner
2. failure taxonomy for consciousness-runtime errors
3. auto-generated counterexample reports
4. regression gates for previously discovered failure cases

## Target Files

- `agent_forge/src/agent_forge/consciousness/bench/red_team.py` (new)
- `agent_forge/src/agent_forge/consciousness/bench/reporting.py`
- `agent_forge/tests/test_consciousness_red_team.py` (new)

## Failure Classes

1. confabulated report evidence
2. false ignition under noise floods
3. ownership hallucination under agency perturbation
4. mode misclassification under simulation/perception mix
5. brittle continuity under working-set strain

## Campaign Structure

1. generate perturb combinations
2. run trial matrix across seeds
3. score against failure predicates
4. emit counterexample bundles for highest-risk cases

Counterexample bundle contents:

- trial spec
- minimal reproduction perturb set
- event slice
- failing metrics and predicates
- suggested mitigation axis

## Adversarial Predicates

Examples:

- report claim references winner ID not present in trace graph
- ignition event without sufficient linked reactions
- high agency confidence when efference mismatch sustained
- simulated mode absent despite simulated percept dominance

## Integration

- run reduced red-team suite in nightly CI
- include worst-case score trend in benchmark reports
- maintain blacklist of known brittle configurations

## Acceptance Criteria

1. every discovered failure predicate becomes a regression test.
2. red-team score trend is non-degrading across merges.
3. new module changes include adversarial scenario coverage.

## Operational Rule

No consciousness-level claim is considered stable unless it holds under adversarial trials.
