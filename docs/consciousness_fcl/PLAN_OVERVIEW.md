# Forge Consciousness Layer (FCL) Plan Overview

## Objective

Implement a modular, falsifiable consciousness-adjacent runtime layer on top of existing Forge primitives:
- append-only event bus
- global workspace broadcast
- self-model snapshots with memory introspection
- MCP tool/resource exposure

The goal is production-grade instrumentation and control, with explicit perturbation and measurable signatures.

## Non-Goals

- No metaphysical claims.
- No opaque "magic" classifier.
- No rewrite of `agent_forge` core runtime.

## Existing Substrate

- Event bus: `agent_forge/src/agent_forge/core/events.py`
- Workspace: `agent_forge/src/agent_forge/core/workspace.py`
- Self-model: `agent_forge/src/agent_forge/core/self_model.py`
- Daemon loop: `agent_forge/src/agent_forge/cli/eidosd.py`
- MCP serving layer: `eidos_mcp/src/eidos_mcp/`

## Functional Spec

The implemented layer must provide:
1. Global availability through workspace broadcasts.
2. Selective bottleneck via explicit candidate competition.
3. Integration and differentiation via perturb-and-measure response complexity metrics.
4. Self-binding via agency and boundary estimates from efference/prediction matching.
5. Grounded reportability via structured self-report events linked to evidence.

## Primary Deliverables

1. `agent_forge/src/agent_forge/consciousness/` package with kernel, modules, metrics, perturbation harness.
2. Event schema standardization and explicit `gw.ignite`/`gw.competition` traces.
3. CLI extensions for workspace/consciousness diagnostics.
4. MCP tools/resources for experimentation and trials.
5. Test suite covering ignition behavior, perturbation metrics, and agency binding behavior.
6. Documentation and operations runbooks for Termux and Linux.

## Milestones

1. Milestone A: minimal GNW loop (attention, competition, ignition, winner observability).
2. Milestone B: policy efference loop and self-binding metrics.
3. Milestone C: perturbation harness and RCI-based trial runner.
4. Milestone D: world-model prediction error, meta-state, and grounded report generation.
5. Milestone E/F/G: benchmark expansion, dynamical continuity, and cross-forge memory/knowledge integration.
6. Milestone H: causal traceability and consciousness-lab instrumentation (schema/index/ignition trace/trial harness/ablations/simulation/PPX/red-team).

## Next Layer Expansion

Detailed Stage H plan documents:

- `docs/consciousness_fcl/NEXT_LAYER_PLAN.md`
- `docs/consciousness_fcl/part-14-north-star-causal-traceability.md`
- `docs/consciousness_fcl/part-15-canonical-schemas-and-indexes.md`
- `docs/consciousness_fcl/part-16-winner-linked-ignition-tracing.md`
- `docs/consciousness_fcl/part-17-consciousness-trial-runner.md`
- `docs/consciousness_fcl/part-18-metrics-v2-connectivity.md`
- `docs/consciousness_fcl/part-19-world-model-v1-5.md`
- `docs/consciousness_fcl/part-20-simulation-stream.md`
- `docs/consciousness_fcl/part-21-phenomenology-probes.md`
- `docs/consciousness_fcl/part-22-perturbation-library-v2.md`
- `docs/consciousness_fcl/part-23-ablation-golden-regression.md`
- `docs/consciousness_fcl/part-24-frontier-upgrades.md`
- `docs/consciousness_fcl/part-25-pr-order-and-gates.md`
- `docs/consciousness_fcl/part-26-adversarial-red-team.md`

## Quality Gates

- All features run in Termux Python venv and standard Linux.
- Deterministic replay for tests where feasible.
- Idempotent command and migration behavior.
- Structured event outputs suitable for offline audit.
- CI-local test pass for changed modules.

## Traceability

Each implementation checkpoint must update `docs/consciousness_fcl/PLAN_TRACKER.md` and mark completed items with strikethrough while retaining full history.

See `docs/consciousness_fcl/REFERENCES.md` for source material.
