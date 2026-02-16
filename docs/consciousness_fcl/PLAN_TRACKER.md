# FCL Plan Tracker

This tracker is the execution control plane for the Forge Consciousness Layer. Completed items are struck through and retained for rollback-safe traceability.

## Status Summary

- Current phase: `Milestone D` complete, benchmark expansion active
- Runtime targets: `Termux` and `Linux`
- Rollback strategy: small commits per vertical slice

## Phase 0: Operational Definition

- [x] ~~Define measurable signatures for global availability, bottlenecking, integration/differentiation, self-binding, and reportability.~~
- [x] ~~Define decision gates for support/falsification based on measured metrics.~~
- [x] ~~Publish event-to-metric mapping spec.~~

## Phase 1: Module Architecture

- [x] ~~Add `agent_forge/src/agent_forge/consciousness/` package skeleton.~~
- [x] ~~Define typed payload contracts and module interfaces.~~
- [x] ~~Add kernel lifecycle and deterministic tick execution.~~

## Phase 2: Data Contracts

- [x] ~~Define event types (`sense.*`, `attn.*`, `gw.*`, `policy.*`, `self.*`, `meta.*`, `report.*`, `perturb.*`, `metrics.*`).~~
- [x] ~~Define canonical workspace payload schema and linkage IDs.~~
- [x] ~~Add schema validation helpers for malformed payload hardening.~~

## Phase 3: Kernel + Runtime Integration

- [x] ~~Integrate consciousness kernel into daemon beat execution path.~~
- [x] ~~Add config loader for module toggles, cadence, and thresholds.~~
- [x] ~~Add safe no-op fallback when optional modules fail.~~

## Phase 4: Core Modules

- [x] ~~`attention` candidate generation.~~
- [x] ~~`workspace_competition` winner selection and ignition marking.~~
- [x] ~~`policy` action/efference loop.~~
- [x] ~~`self_model_ext` agency/boundary estimates.~~
- [x] ~~`world_model` prediction and error streams.~~
- [x] ~~`meta` mode estimation.~~
- [x] ~~`report` grounded self-reporting.~~

## Phase 5: Perturb and Measure

- [x] ~~Add perturbation library and harness.~~
- [x] ~~Add response complexity metric (RCI-like) and supporting metrics.~~
- [x] ~~Add standardized trial runner and report persistence.~~

## Phase 6: CLI and MCP

- [x] ~~Extend `eidctl workspace` with winners/coherence/RCI/agency views.~~
- [x] ~~Add `eidctl consciousness` commands for status and trials.~~
- [x] ~~Expose MCP tools/resources for consciousness status, perturbation, and trial execution.~~
- [x] ~~Add `eidctl consciousness benchmark` and `latest-benchmark`.~~
- [x] ~~Expose MCP benchmark tools/resources for runtime benchmark reports.~~

## Phase 7: Tests and Validation

- [x] ~~Add unit tests for attention/competition/ignition.~~
- [x] ~~Add perturbation-response metric tests.~~
- [x] ~~Add agency-binding falsification tests.~~
- [x] ~~Add integration test for daemon + kernel + event log outputs.~~
- [x] ~~Add Stage D tests for world/model/meta/report event behavior.~~
- [x] ~~Add benchmark suite tests for baseline delta and CLI path.~~

## Phase 8: Termux and Linux Hardening

- [x] ~~Verify venv compatibility and import path stability in Termux.~~
- [x] ~~Add defensive filesystem handling for Android/Termux constraints.~~
- [x] ~~Add robust URL/TLS fallback handling for Termux Tika URL ingestion paths.~~
- [ ] Verify parity behavior in Linux runner.

## Phase 9: Documentation and Operations

- [x] ~~Add architecture and runbook docs.~~
- [x] ~~Add troubleshooting matrix and known failure modes.~~
- [x] ~~Add reproducible benchmark/audit commands.~~

## Phase 10: Claims and Level Gates

- [x] ~~Map level labels to measurable signatures only.~~
- [x] ~~Add acceptance/failure criteria per level.~~
- [x] ~~Add explicit disconfirmation criteria and reporting.~~

## Phase 11: Continuous Improvement Benchmarking

- [x] ~~Add internal benchmark suite with latency, capability, and gate checks.~~
- [x] ~~Add baseline comparison and non-regression gates.~~
- [x] ~~Add optional external benchmark score ingestion + normalization.~~
- [ ] Add automated Linux parity benchmark job and publish trend reports.
