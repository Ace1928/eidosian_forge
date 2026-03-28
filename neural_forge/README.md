# Neural Forge

## Purpose

`neural_forge` is the experimental neural-substrate package for Eidosian model research.
It is not yet the production inference backbone for the forge. Right now it is a prototype PyTorch research surface with working low-level components, model-level smoke paths, and early training harness coverage.

## What Exists Today

### Core substrate
- ternary-weight `BitLinear` and `BitRMSNorm` in `src/neural_forge/core/layers.py`
- prototype mixture-of-experts routing with a hyper-expert path in `src/neural_forge/core/moe.py`
- modality classifier and shared latent projection layers in `src/neural_forge/core/modality.py`

### Model assembly
- `BitTransformer` baseline in `src/neural_forge/models/backbone.py`
- `NeuralForgeSingularity` backbone in `src/neural_forge/models/singularity.py`
- `EidosianSingularityModel` wrapper in `src/neural_forge/models/integrated.py`

### Training harness
- minimal training step harness in `src/neural_forge/training/trainer.py`

### Validation
Current tests cover:
- core layers
- modality routing
- MoE primitives
- model smoke integration
- trainer smoke execution

## What It Does Well

- keeps the neural-substrate work modular and isolated from the production runtime
- provides explicit, testable primitives for quantized layers and sparse routing research
- now supports both token-id text input and precomputed feature input for explicit modalities

## Current Limits

- this is still prototype-grade research code, not a deployable training stack
- image/audio support currently expects precomputed features rather than full encoders
- sparse routing is implemented in Python loops and is not performance-tuned
- no checkpoint, dataset, or benchmark contract is complete yet

## Why It Exists

The point of Neural Forge is to give the forge a place to evolve substrate-level learning and representation work without polluting the operational control plane. It is where architecture experiments should become measurable code before any deeper integration claims are made.

## Directory Map

- `src/neural_forge/core/`: substrate primitives
- `src/neural_forge/models/`: assembled backbones and integrated wrappers
- `src/neural_forge/training/`: training harnesses
- `tests/`: validation

## Validation

Run the focused slice:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=neural_forge/src ./eidosian_venv/bin/python -m pytest -q \
  neural_forge/tests/test_layers.py \
  neural_forge/tests/test_modality.py \
  neural_forge/tests/test_moe.py \
  neural_forge/tests/test_models.py \
  neural_forge/tests/test_trainer.py
```

## Next Steps

1. add checkpoint and serialization contracts
2. replace prototype MoE routing with a real sparse execution path
3. add explicit encoder modules and benchmark CPU memory/latency on Termux and Linux
