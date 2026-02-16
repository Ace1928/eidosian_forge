# FCL Plan Tracker

This tracker is the execution control plane for the Forge Consciousness Layer. Completed items are struck through and retained for rollback-safe traceability.

## Status Summary

- Current phase: `Milestone I` active (self-upgrading loop substrate + safe autotuning)
- Runtime targets: `Termux` and `Linux`
- Rollback strategy: small commits per vertical slice
- MCP audit status (February 16, 2026): tools `122/123` ok with `1` intentional skip (`mcp_self_upgrade`), resources `11/11` ok.

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
- [x] ~~Add `eidctl consciousness full-benchmark` and `latest-full-benchmark`.~~
- [x] ~~Expose MCP integrated benchmark tool/resource (`consciousness_kernel_full_benchmark`, `consciousness_kernel_latest_full_benchmark`, `eidos://consciousness/runtime-latest-full-benchmark`).~~

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
- [x] ~~Add Linux parity smoke script and CI workflow (`scripts/linux_parity_smoke.sh`, `.github/workflows/consciousness-parity.yml`).~~
- [x] ~~Verify parity behavior in Linux runner.~~
- [x] ~~Harden full MCP integration tests to backup/restore KB and memory artifacts to prevent repo-state pollution.~~

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
- [x] ~~Add integrated stack benchmark (kernel + trials + optional MCP + optional local LLM) with persisted reports and baseline delta.~~
- [x] ~~Add automated Linux parity benchmark job and publish trend reports.~~
- [x] ~~Integrate adversarial red-team scoring/gates into integrated benchmark runtime and exposed CLI/MCP surfaces.~~

## Phase 12: Dynamical Continuity and Emergence

- [x] ~~Add persistent per-module state store (`state_store.py`) with periodic checkpoints.~~
- [x] ~~Add multi-rate module scheduling (`module_tick_periods`) and module disable controls.~~
- [x] ~~Implement stateful `sense`, `intero`, and `affect` modules and wire into default kernel path.~~
- [x] ~~Implement `working_set` continuity module with decay/capacity and `WM_STATE` broadcast.~~
- [x] ~~Upgrade competition to emit winner-linked `gw.reaction_trace` and stronger ignition criteria.~~
- [x] ~~Promote perturbation runtime to active module-aware behavior (`drop`, `noise`, `delay`, `clamp`, `scramble`).~~
- [x] ~~Add continuity regression tests (`test_consciousness_continuity.py`).~~
- [x] ~~Extend world-model to latent feature predictive coding (beyond event-type transitions).~~
- [x] ~~Add explicit memory and knowledge bridge modules (`memory_forge` + `knowledge_forge` -> recalls/context/events).~~
- [x] ~~Expose bridge integration status through MCP (`consciousness_bridge_status`, `eidos://consciousness/runtime-integrations`).~~
- [x] ~~Add ablation benchmark runner and contribution assertions.~~

## Phase 13: Causal Instrumentation and Experimental Rigor

- [x] ~~Publish next-layer execution plan documents (`part-14` through `part-26`).~~
- [x] ~~Enforce canonical link contract (`corr_id`, `parent_id`, `winner_candidate_id`, `candidate_id`) across scored workspace events.~~
- [x] ~~Add beat-local `EventIndex` foundation and TickContext index helpers.~~
- [x] ~~Migrate critical runtime modules to index-based lookups (`workspace_competition`, `report`).~~
- [x] ~~Migrate benchmark extraction paths to index-based lookups.~~
- [x] ~~Implement winner-linked ignition tracing metric (`trace_strength`) and ignition v3 gates.~~
- [x] ~~Add standardized consciousness trial harness package (`bench/`) with persisted artifacts (`spec.json`, `metrics.jsonl`, `events_window.jsonl`, `summary.md`).~~
- [x] ~~Upgrade metrics suite to v2 (`entropy`, `connectivity`, `directionality`, `self_stability`).~~
- [x] ~~Upgrade world model to belief-state predictive coding with rollout API.~~
- [x] ~~Add simulation stream module and mode-aware report/meta integration.~~
- [x] ~~Add phenomenology probe module (PPX indices + snapshot events).~~
- [x] ~~Add perturbation library v2 composite recipes with expected signatures.~~
- [x] ~~Add ablation matrix, golden ranges, and regression gates (initial runner + checks + tests).~~
- [x] ~~Add adversarial self red-team campaigns and counterexample bundles.~~

## Phase 14: Self-Upgrading Consciousness Loops

- [x] ~~Add canonical parameter spec layer (`tuning/params.py`) with safety/range metadata.~~
- [x] ~~Add tuned overlay sanitation + persistence + versioned history (`tuning/overlay.py`).~~
- [x] ~~Integrate base+tuned+runtime config resolution in kernel (`resolve_config` + runtime overrides).~~
- [x] ~~Extend trial runner with overlay-aware execution and explicit guardrail metrics (`module_error_count`, `degraded_mode_ratio`, `winner_count`, `ignitions_without_trace`).~~
- [x] ~~Add bootstrap autotune runtime module with propose→trial→commit/rollback loop (`modules/autotune.py`).~~
- [x] ~~Upgrade optimizer from bandit hill-climbing to Bayesian/multi-objective strategy (`bayes_pareto` acquisition + frontier tracking).~~
- [x] ~~Add adaptive attention/competition weights learned from trial objective and ignition trace feedback.~~
- [x] ~~Add closed-loop micro-task suite (`self_other_discrimination`, continuity under distraction, report-grounding challenge).~~
- [x] ~~Add experiment-designer module for safe self-generated perturbation campaigns.~~
- [x] ~~Add adversarial tuner red-team gates and regression bundles (red-team campaign + runtime CLI/MCP surfaces).~~
- [x] ~~Integrate red-team outputs into full-benchmark composite scoring and hard gate checks.~~
- [x] ~~Gate autotune overlay commits behind quick red-team thresholds with availability safeguards and explicit rollback reason tracing.~~

## Phase 15: Workflow and Supply-Chain Hardening

- [x] ~~Add scheduled/manual security audit workflow for Dependabot inventory with optional critical/high fail gates and artifacted reports.~~
- [x] ~~Add automated remediation issue generation from security inventory deltas.~~
- [ ] Add pinned-action drift audit and update policy enforcement in CI.
