# Neural Forge Training

## Scope

This directory holds the training harness for Neural Forge prototypes.

## Implemented Components

- `trainer.py`: a minimal trainer using AdamW, cross-entropy loss, and gradient clipping

## Current Strengths

- enough to run deterministic smoke training on the text token path
- keeps the training contract explicit and testable

## Current Limits

- this is not yet GaLore, Sophia, or a full large-scale training stack
- no dataset pipeline, checkpoint manager, or mixed-precision policy is implemented here

## Validation

Trainer smoke coverage lives in `neural_forge/tests/test_trainer.py`.

## Next Steps

1. add checkpoint save/load contracts
2. add pluggable optimizer policies
3. add reproducible benchmark scripts for CPU edge training
