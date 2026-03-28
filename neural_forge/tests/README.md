# Neural Forge Tests

## Scope

This directory validates the currently implemented Neural Forge substrate.

## Covered Today

- low-level layers
- modality routing
- mixture-of-experts primitives
- model-level integration smoke tests
- trainer smoke behavior

## Current Gaps

- no checkpoint round-trip tests yet
- no performance benchmark suite yet
- no long-run training convergence tests yet

## Next Steps

1. add checkpoint and serialization tests
2. add CPU memory/latency benchmarks
3. add regression fixtures for future encoder implementations
