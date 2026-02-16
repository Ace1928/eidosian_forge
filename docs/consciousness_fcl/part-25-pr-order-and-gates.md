# Part 25: PR Order and Release Gates

## Goal

Convert the roadmap into concrete PR sequencing with objective completion gates.

## PR Sequence

1. PR-H1: canonical links + index foundation
2. PR-H2: winner-linked ignition tracing
3. PR-H3: standardized trial runner
4. PR-H4: metrics v2 and connectivity scoring
5. PR-H5: ablation matrix + golden ranges
6. PR-H6: world model v1.5
7. PR-H7: simulation stream
8. PR-H8: phenomenology probes
9. PR-H9: perturbation library v2 recipes
10. PR-H10: frontier upgrade set (flagged)
11. PR-H11: adversarial red-team suite

## PR Template Requirements

Each PR must include:

- schema updates (if event shape changed)
- module/API changes
- tests (unit + integration)
- benchmark evidence (before/after)
- rollback note

## Gate Classes

### Gate A: Functional

- relevant tests pass
- no hard failures in MCP tool/resource audit

### Gate B: Behavioral

- expected signatures observed in targeted trials
- non-regression against last baseline report

### Gate C: Performance

- tick latency p95 remains within threshold
- event volume growth remains controlled

### Gate D: Reproducibility

- deterministic seed run variability within bounds
- artifact persistence and replay checks pass

## Promotion Policy

- merge only if all four gates pass for PR scope
- emergency override requires explicit documented exception and follow-up issue

## Release Cadence

- micro-PR cadence preferred (one technical axis per PR)
- periodic integration PR consolidates docs and benchmark trend snapshots

## Acceptance Criteria

1. execution order maintained unless a documented dependency inversion is required.
2. every merged PR includes measured evidence, not only implementation claims.
3. benchmark drift is tracked continuously, not per release only.
