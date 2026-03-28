# Neural Forge Models

## Scope

This directory assembles the core primitives into model-level backbones.

## Implemented Components

- `backbone.py`: BitTransformer-style baseline blocks using `BitLinear`
- `singularity.py`: backbone with BitAttention plus BitMoE blocks
- `integrated.py`: top-level multimodal wrapper and conflict monitor

## Current Strengths

- token-id text path is functional
- pre-embedded feature path now works through `forward_embedded()`
- explicit modality handling avoids silent shape misuse

## Current Limits

- image/audio paths assume precomputed features instead of full encoders
- automatic modality inference is only safe for shared-width feature tensors
- no training-scale benchmark or checkpoint contract exists yet

## Validation

Model integration is covered by `neural_forge/tests/test_models.py` plus the core unit tests.

## Next Steps

1. add encoder modules for image and audio instead of relying on precomputed features
2. add checkpoint and serialization tests
3. connect conflict-monitor outputs into consciousness/runtime evidence paths
