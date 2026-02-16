# Part 07: Tests and Validation

## Goal

Guarantee behavior and regressions are detectable.

## Test Matrix

1. attention scoring determinism
2. competition winner selection and ignition event emission
3. perturbation effect on response metrics
4. agency confidence drop on efference mismatch
5. daemon integration with kernel enabled

## Acceptance Criteria

- All new tests pass on Termux and Linux.
- Test fixtures are idempotent and isolated.
