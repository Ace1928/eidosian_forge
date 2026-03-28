# Neural Forge Core

## Scope

This directory contains the low-level neural substrate primitives currently implemented for Neural Forge.

## Implemented Components

- `layers.py`: ternary-weight `BitLinear`, activation quantization, and `BitRMSNorm`
- `moe.py`: prototype sparse expert routing with a hyper-expert path
- `modality.py`: modality enum, classifier, and shared latent projections
- `tokenizer.py`: tokenizer utilities for the text path

## Current Strengths

- small, testable PyTorch components
- explicit quantization and routing primitives
- enough structure to support model-level smoke tests

## Current Limits

- routing is prototype-grade and uses per-batch Python loops in the MoE path
- no production sparse-kernel backend exists yet
- modality routing is still a projection layer, not a full encoder stack

## Validation

Current validating tests live in `neural_forge/tests/` and cover layers, modality, MoE, model integration, and trainer smoke behavior.

## Next Steps

1. replace the prototype MoE batch loop with a real sparse execution path
2. add modality-specific encoder implementations behind the shared latent contract
3. benchmark CPU memory and latency on Termux and Linux
