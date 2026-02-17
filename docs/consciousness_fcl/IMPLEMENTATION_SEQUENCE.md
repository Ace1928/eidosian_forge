# Detailed Implementation Sequence

This sequence is commit-sized, idempotent, and rollback-friendly. Each step includes target files, validation commands, and rollback notes.

## Stage A (Completed): Minimal GNW Loop

1. Package + contracts
- Files: `agent_forge/src/agent_forge/consciousness/types.py`, `agent_forge/src/agent_forge/consciousness/kernel.py`
- Output: deterministic `TickContext`, payload normalization, module protocol, kernel result contract.
- Validation: `tests/test_consciousness_milestone_a.py::test_attention_emits_candidates`

2. Attention candidates
- Files: `agent_forge/src/agent_forge/consciousness/modules/attention.py`
- Output: `attn.candidate` events with scores and source links.
- Validation: `tests/test_consciousness_milestone_a.py::test_attention_emits_candidates`

3. Competition + ignition
- Files: `agent_forge/src/agent_forge/consciousness/modules/workspace_competition.py`
- Output: `gw.competition`, winner broadcast packets, `gw.ignite` marker.
- Validation: `tests/test_consciousness_milestone_a.py::test_competition_broadcasts_winner_and_ignite`

4. Runtime wiring
- Files: `agent_forge/src/agent_forge/cli/eidosd.py`
- Output: optional kernel tick each daemon beat, `consciousness.beat` event and metric.
- Validation: `tests/test_consciousness_milestone_a.py::test_eidosd_once_emits_consciousness_beat`

5. Observability
- Files: `agent_forge/src/agent_forge/cli/eidctl.py`
- Output: `eidctl workspace --show-winners --show-coherence --show-rci --show-agency`
- Validation: manual CLI run + workspace event inspection.

6. Termux hardening fixes discovered during implementation
- Files: `agent_forge/src/agent_forge/core/os_metrics.py`, `agent_forge/src/agent_forge/agent_core.py`, `agent_forge/src/agent_forge/__init__.py`
- Output:
  - `os.getloadavg` graceful fallback in Termux.
  - defensive `llm_forge` import handling to avoid `AgentForge=None` in MCP state.
  - safe package import behavior for partial startup.
  - full MCP integration test harness now backs up/restores mutable files (`data/kb.json`, `memory_data.json`, tiered memory JSONs, semantic memory) to keep repo state idempotent.
- Validation:
  - `eidos_mcp.state` now resolves `AgentForge` and non-`None` `agent`.
  - daemon tests pass in Termux.

## Stage B (Completed): Self-Binding Loop

1. Add `policy` module for action/efference emission.
- Files: `agent_forge/src/agent_forge/consciousness/modules/policy.py`
2. Add `self_model_ext` module for agency/boundary confidence.
- Files: `agent_forge/src/agent_forge/consciousness/modules/self_model_ext.py`
3. Extend self snapshot output in `agent_forge/src/agent_forge/core/self_model.py`.
4. Add falsification tests:
- `agent_forge/tests/test_consciousness_milestone_b.py::test_self_model_ext_reduces_agency_on_efference_mismatch`
- `agent_forge/tests/test_consciousness_milestone_b.py::test_self_model_ext_emits_agency_and_boundary`

## Stage C (Completed): Perturb and Measure

1. Expand perturb harness to active injection adapters.
2. Add trial runner with fixed perturbation protocols and persisted reports.
- Files: `agent_forge/src/agent_forge/consciousness/trials.py`
3. Persist RCI and supporting features into event + metrics stores.
4. Add CLI command group `eidctl consciousness trial/status`.
5. Add MCP tools for perturb/trial/export trace.
- Files: `eidos_mcp/src/eidos_mcp/routers/consciousness.py`
- Validation:
  - `agent_forge/tests/test_consciousness_trials.py`
  - `eidos_mcp/tests/test_mcp_tool_calls_individual.py`

## Stage D (Completed): World/Meta/Report

1. Implement predictive world model and prediction error streams.
- Files: `agent_forge/src/agent_forge/consciousness/modules/world_model.py`
2. Implement `meta` state mode classifier from measured dynamics.
- Files: `agent_forge/src/agent_forge/consciousness/modules/meta.py`
3. Implement grounded `report` module with explicit disconfirmers.
- Files: `agent_forge/src/agent_forge/consciousness/modules/report.py`
4. Add calibration tests for report confidence quality.
- Files: `agent_forge/tests/test_consciousness_milestone_d.py`

## Stage E (Completed): Continuous Benchmark Expansion

1. Add internal benchmark suite with reproducible scoring.
- Files: `agent_forge/src/agent_forge/consciousness/benchmarks.py`
2. Expose benchmark controls through CLI and MCP.
- Files: `agent_forge/src/agent_forge/cli/eidctl.py`
- Files: `eidos_mcp/src/eidos_mcp/routers/consciousness.py`
3. Add benchmark tests and baseline-delta checks.
- Files: `agent_forge/tests/test_consciousness_benchmarks.py`
4. Add external benchmark score ingestion adapters and normalization.
5. Add integrated stack benchmark and MCP accessors.
- Files: `agent_forge/src/agent_forge/consciousness/integrated_benchmark.py`
- Files: `agent_forge/tests/test_consciousness_integrated_benchmark.py`
- Files: `eidos_mcp/src/eidos_mcp/routers/consciousness.py`
- Files: `eidos_mcp/tests/test_mcp_tool_calls_individual.py`
6. Add automated Linux parity benchmark trend job.
- Files: `.github/workflows/consciousness-parity.yml`
- Files: `scripts/consciousness_benchmark_trend.py`
- Files: `scripts/tests/test_consciousness_benchmark_trend.py`

## Stage F (Completed): Dynamical Continuity and Multi-Timescale Runtime

1. Add persistent module state store and kernel beat persistence.
- Files: `agent_forge/src/agent_forge/consciousness/state_store.py`
- Files: `agent_forge/src/agent_forge/consciousness/kernel.py`

2. Extend `TickContext` for indexed access and perturb querying.
- Files: `agent_forge/src/agent_forge/consciousness/types.py`

3. Wire substrate modules into default execution path.
- Files: `agent_forge/src/agent_forge/consciousness/modules/sense.py`
- Files: `agent_forge/src/agent_forge/consciousness/modules/intero.py`
- Files: `agent_forge/src/agent_forge/consciousness/modules/affect.py`
- Files: `agent_forge/src/agent_forge/consciousness/modules/working_set.py`
- Files: `agent_forge/src/agent_forge/consciousness/modules/__init__.py`

4. Improve GNW quality with winner-linked reaction traces and inhibition-of-return.
- Files: `agent_forge/src/agent_forge/consciousness/modules/workspace_competition.py`

5. Promote perturb harness from no-op adapter to active kernel registration.
- Files: `agent_forge/src/agent_forge/consciousness/perturb/harness.py`

6. Add continuity and cadence regression tests.
- Files: `agent_forge/tests/test_consciousness_continuity.py`

## Stage G (Completed): Cross-Forge Memory/Knowledge Integration

1. Add memory bridge module with optional `memory_forge` introspection/recall.
- Files: `agent_forge/src/agent_forge/consciousness/modules/memory_bridge.py`

2. Add knowledge bridge module with optional `knowledge_forge` unified context lookup.
- Files: `agent_forge/src/agent_forge/consciousness/modules/knowledge_bridge.py`

3. Wire bridge modules into default kernel path and module cadence defaults.
- Files: `agent_forge/src/agent_forge/consciousness/kernel.py`
- Files: `agent_forge/src/agent_forge/consciousness/types.py`
- Files: `agent_forge/src/agent_forge/consciousness/modules/__init__.py`

4. Expose bridge state in runtime snapshots and benchmark surfaces.
- Files: `agent_forge/src/agent_forge/consciousness/trials.py`
- Files: `agent_forge/src/agent_forge/consciousness/benchmarks.py`
- Files: `agent_forge/src/agent_forge/core/self_model.py`

5. Expose bridge status through MCP tool/resource.
- Files: `eidos_mcp/src/eidos_mcp/routers/consciousness.py`

6. Add bridge integration regression tests and parity matrix coverage.
- Files: `agent_forge/tests/test_consciousness_memory_knowledge_bridge.py`
- Files: `eidos_mcp/tests/test_mcp_tool_calls_individual.py`
- Files: `scripts/linux_parity_smoke.sh`

## Stage H (Completed): Causal Traceability and Consciousness-Lab Instrumentation

Status checkpoint:
- PR-H1 delivered (canonical links + `EventIndex` + `TickContext` index/link helpers + indexed `workspace_competition`/`report` lookups + regression tests).
- PR-H2 delivered (winner-linked `trace_strength` instrumentation + ignition v3 gates + pending winner finalization + regressions).
- PR-H3 delivered (new `consciousness/bench/` package with `TrialSpec`, stage lifecycle execution, artifact persistence, and regression tests).
- PR-H4 delivered (metrics v2 modules: `entropy`, `connectivity`, `directionality`, `self_stability`, plus upgraded `rci_v2` and bench integration).
- PR-H5 delivered (ablation matrix runner, golden checks, and regression tests).
- PR-H6 delivered (world model v1.5 predictive coding with feature extraction, persistent belief state, surprise decomposition, and rollout API).
- PR-H7 delivered (simulation stream module + meta/report simulated-state integration + regression tests).
- PR-H8 delivered (phenomenology probe module with PPX indices, runtime/status integration, and scoring deltas).
- PR-H9 delivered (perturbation recipe library v2, recipe-aware trial expansion, expected-signature checks, and module-level perturb hooks).
- PR-H10 delivered (benchmark extraction path migration to `EventIndex` for snapshot and trial window counters).

1. Canonical link contract and schema hardening (PR-H1).
- Files: `agent_forge/src/agent_forge/consciousness/types.py`
- Files: `agent_forge/src/agent_forge/consciousness/linking.py` (new)
- Files: `agent_forge/src/agent_forge/consciousness/schemas.py` (new)

2. Event index foundation and indexed module migration (PR-H1 continuation).
- Files: `agent_forge/src/agent_forge/consciousness/index.py` (new)
- Files: `agent_forge/src/agent_forge/consciousness/types.py`
- Files: `agent_forge/src/agent_forge/consciousness/modules/workspace_competition.py`
- Files: `agent_forge/src/agent_forge/consciousness/modules/report.py`

3. Winner-linked ignition tracing and ignition v3 gates (PR-H2).
- Files: `agent_forge/src/agent_forge/consciousness/metrics/ignition_trace.py` (new)
- Files: `agent_forge/src/agent_forge/consciousness/modules/workspace_competition.py`

4. Standardized CTR package for reproducible trials (PR-H3).
- Files: `agent_forge/src/agent_forge/consciousness/bench/trials.py` (new)
- Files: `agent_forge/src/agent_forge/consciousness/bench/tasks.py` (new)
- Files: `agent_forge/src/agent_forge/consciousness/bench/scoring.py` (new)
- Files: `agent_forge/src/agent_forge/consciousness/bench/reporting.py` (new)

5. Metrics v2 and connectivity instrumentation (PR-H4).
- Files: `agent_forge/src/agent_forge/consciousness/metrics/entropy.py` (new)
- Files: `agent_forge/src/agent_forge/consciousness/metrics/connectivity.py` (new)
- Files: `agent_forge/src/agent_forge/consciousness/metrics/self_stability.py` (new)
- Files: `agent_forge/src/agent_forge/consciousness/metrics/rci.py`

6. Ablation matrix and golden ranges (PR-H5).
- Files: `agent_forge/src/agent_forge/consciousness/bench/ablations.py` (new)
- Files: `agent_forge/src/agent_forge/consciousness/bench/golden.py` (new)
- Files: `agent_forge/tests/test_consciousness_ablations.py` (new)

7. World model v1.5 predictive coding and rollout API (PR-H6).
- Files: `agent_forge/src/agent_forge/consciousness/modules/world_model.py`
- Files: `agent_forge/src/agent_forge/consciousness/features.py` (new)

8. Simulation stream integration (PR-H7).
- Files: `agent_forge/src/agent_forge/consciousness/modules/simulation.py` (new)
- Files: `agent_forge/src/agent_forge/consciousness/modules/meta.py`
- Files: `agent_forge/src/agent_forge/consciousness/modules/report.py`

9. Phenomenology probes and PPX indices (PR-H8).
- Files: `agent_forge/src/agent_forge/consciousness/modules/phenomenology_probe.py` (new)
- Files: `agent_forge/src/agent_forge/consciousness/bench/scoring.py`

10. Perturbation library v2 recipes (PR-H9).
- Files: `agent_forge/src/agent_forge/consciousness/perturb/library.py`
- Files: `agent_forge/src/agent_forge/consciousness/perturb/harness.py`

11. Adversarial red-team campaigns (PR-H11).
- Files: `agent_forge/src/agent_forge/consciousness/bench/red_team.py` (new)
- Files: `agent_forge/tests/test_consciousness_red_team.py` (new)

12. Benchmark extraction migration to index-backed lookups (PR-H10).
- Files: `agent_forge/src/agent_forge/consciousness/index.py`
- Files: `agent_forge/src/agent_forge/consciousness/bench/trials.py`
- Files: `agent_forge/tests/test_consciousness_linking_index.py`

## Validation Commands

1. Agent Forge targeted tests:
```sh
PYTHONPATH=/data/data/com.termux/files/home/eidosian_forge/lib:/data/data/com.termux/files/home/eidosian_forge/agent_forge/src ../eidosian_venv/bin/python -m pytest -q tests/test_consciousness_milestone_a.py tests/test_workspace.py tests/test_db_and_daemon.py
```
1.1 Continuity tests:
```sh
PYTHONPATH=/data/data/com.termux/files/home/eidosian_forge/lib:/data/data/com.termux/files/home/eidosian_forge/agent_forge/src:/data/data/com.termux/files/home/eidosian_forge/eidos_mcp/src:/data/data/com.termux/files/home/eidosian_forge/crawl_forge/src eidosian_venv/bin/python -m pytest -q agent_forge/tests/test_consciousness_continuity.py
```
2. MCP tests:
```sh
PYTHONPATH=/data/data/com.termux/files/home/eidosian_forge/lib:/data/data/com.termux/files/home/eidosian_forge/eidos_mcp/src:/data/data/com.termux/files/home/eidosian_forge/agent_forge/src:/data/data/com.termux/files/home/eidosian_forge eidosian_venv/bin/python -m pytest -q eidos_mcp/tests/test_mcp_tool_calls_individual.py eidos_mcp/tests/test_diagnostics_transport_matrix.py
PYTHONPATH=/data/data/com.termux/files/home/eidosian_forge/lib:/data/data/com.termux/files/home/eidosian_forge/eidos_mcp/src:/data/data/com.termux/files/home/eidosian_forge/agent_forge/src:/data/data/com.termux/files/home/eidosian_forge EIDOS_RUN_FULL_INTEGRATION=1 eidosian_venv/bin/python -m pytest -q eidos_mcp/tests/test_mcp_tools_stdio.py
```
3. Audits:
```sh
./scripts/termux_audit.sh
PYTHONPATH=/data/data/com.termux/files/home/eidosian_forge/lib:/data/data/com.termux/files/home/eidosian_forge/eidos_mcp/src:/data/data/com.termux/files/home/eidosian_forge/agent_forge/src:/data/data/com.termux/files/home/eidosian_forge eidosian_venv/bin/python scripts/audit_mcp_tools_resources.py --timeout 8
```

## Rollback Guidance

- Revert by commit boundaries only.
- Preserve runtime data snapshots and transaction logs.
- Avoid hard reset in shared workflows.

## Stage I (Completed): Self-Upgrading Loop Substrate

Status checkpoint:
- PR-I1 delivered (parameter specs + overlay sanitation/persistence + kernel config precedence).
- PR-I2 delivered (overlay-aware trial execution + guardrail counters in trial reports).
- PR-I3 delivered (bootstrap `autotune` module with propose/trial/commit/rollback loop and metrics).
- PR-I4 delivered (Bayesian/multi-objective optimizer path with Pareto frontier tracking and acquisition scoring).
- PR-I5 delivered (adaptive attention and competition policy learning loops driven by ignition trace feedback).
- PR-I6 delivered (experiment-designer runtime module + red-team campaign harness + CLI/MCP integration + regression tests).
- PR-I7 delivered (integrated benchmark red-team scoring + gate integration + CLI/MCP full-benchmark controls).
- PR-I8 delivered (autotune commit-path red-team gate with configurable pass/robustness thresholds and availability-safe rollback semantics).

1. Parameter control plane and safety classes (PR-I1).
- Files: `agent_forge/src/agent_forge/consciousness/tuning/params.py`
- Files: `agent_forge/src/agent_forge/consciousness/tuning/overlay.py`

2. Kernel config precedence and runtime overrides (PR-I1).
- Files: `agent_forge/src/agent_forge/consciousness/kernel.py`
- Files: `agent_forge/src/agent_forge/consciousness/types.py`

3. Trial runner overlay execution + explicit guardrail metrics (PR-I2).
- Files: `agent_forge/src/agent_forge/consciousness/bench/trials.py`

4. Autotune runtime module (PR-I3).
- Files: `agent_forge/src/agent_forge/consciousness/modules/autotune.py`
- Files: `agent_forge/src/agent_forge/consciousness/tuning/optimizer.py`

5. Bayesian/multi-objective optimizer path (PR-I4).
- Files: `agent_forge/src/agent_forge/consciousness/tuning/bayes_optimizer.py`
- Files: `agent_forge/src/agent_forge/consciousness/tuning/objectives.py`
- Files: `agent_forge/src/agent_forge/consciousness/modules/autotune.py`

6. Adaptive attention/competition learning (PR-I5).
- Files: `agent_forge/src/agent_forge/consciousness/modules/attention.py`
- Files: `agent_forge/src/agent_forge/consciousness/modules/workspace_competition.py`
- Files: `agent_forge/tests/test_consciousness_attention_competition_learning.py`

7. Experiment-designer runtime module and safe perturb proposal loop (PR-I6).
- Files: `agent_forge/src/agent_forge/consciousness/modules/experiment_designer.py`
- Files: `agent_forge/src/agent_forge/consciousness/kernel.py`
- Files: `agent_forge/src/agent_forge/consciousness/types.py`

8. Adversarial red-team campaigns and integration surfaces (PR-I6).
- Files: `agent_forge/src/agent_forge/consciousness/bench/red_team.py`
- Files: `agent_forge/src/agent_forge/consciousness/bench/tasks.py`
- Files: `agent_forge/src/agent_forge/cli/eidctl.py`
- Files: `eidos_mcp/src/eidos_mcp/routers/consciousness.py`
- Files: `agent_forge/tests/test_consciousness_red_team.py`
- Files: `agent_forge/tests/test_consciousness_experiment_designer.py`

9. Integrated full-benchmark red-team gates and weighted scoring normalization (PR-I7).
- Files: `agent_forge/src/agent_forge/consciousness/integrated_benchmark.py`
- Files: `agent_forge/src/agent_forge/cli/eidctl.py`
- Files: `eidos_mcp/src/eidos_mcp/routers/consciousness.py`
- Files: `agent_forge/tests/test_consciousness_integrated_benchmark.py`
- Files: `eidos_mcp/tests/test_mcp_tool_calls_individual.py`

10. Autotune commit-path red-team guard and rollback reason tracing (PR-I8).
- Files: `agent_forge/src/agent_forge/consciousness/modules/autotune.py`
- Files: `agent_forge/src/agent_forge/consciousness/types.py`
- Files: `agent_forge/tests/test_consciousness_autotune.py`
- Files: `agent_forge/tests/test_consciousness_red_team.py`

## Stage J (Completed): Workflow and Supply-Chain Hardening

Status checkpoint:
- PR-J1 delivered (scheduled/manual security audit workflow using Dependabot inventory script, optional severity hard-fail gates, and artifacted reports).
- PR-J2 delivered (automated delta-to-issue remediation loop with baseline markers and close-on-clear semantics).
- PR-J3 delivered (workflow action pin drift audit script + lock file + CI enforcement workflow).

1. Dependency security inventory workflow (PR-J1).
- Files: `.github/workflows/security-audit.yml`
- Files: `scripts/dependabot_alert_inventory.py`
- Files: `.github/workflows/README.md`

2. Security inventory delta remediation issues (PR-J2).
- Files: `.github/workflows/security-audit.yml`
- Files: `docs/consciousness_fcl/part-36-security-delta-remediation-issues.md`

3. Workflow action pin drift policy enforcement (PR-J3).
- Files: `scripts/audit_workflow_action_pins.py`
- Files: `.github/workflows/workflow-action-pin-audit.yml`
- Files: `audit_data/workflow_action_lock.json`
- Files: `scripts/tests/test_audit_workflow_action_pins.py`

## Stage K (Completed): Vulnerability Remediation Orchestration

Status checkpoint:
- PR-K1 delivered (Dependabot remediation batching planner, issue sync engine, workflow integration, and regression tests).

1. Remediation batch planner + issue synchronization (PR-K1).
- Files: `scripts/dependabot_remediation_plan.py`
- Files: `scripts/sync_security_remediation_issues.py`
- Files: `scripts/tests/test_dependabot_remediation_plan.py`
- Files: `scripts/tests/test_sync_security_remediation_issues.py`
- Files: `.github/workflows/security-audit.yml`

## Stage L (Completed): Deterministic Dependency Auto-Patch Execution

Status checkpoint:
- PR-L1 delivered (raw-alert-driven pip requirement auto-patch tool with dry-run/write + rollback backups + tests).
- PR-L2 delivered (`security-audit.yml` now exports raw alerts and publishes autopatch dry-run summary/artifacts).
- PR-L3 delivered (high/critical write pass across four pip manifests with post-write idempotency verification).
- PR-L4 delivered (all-severity write pass with deterministic pin upgrades and post-write idempotency verification).
- PR-L5 delivered (no-fix advisory mitigation via direct dependency minimization and manual vulnerable-range escape for `orjson`).
- PR-L6 delivered (Dependabot refresh converged to zero open alerts; remediation issue sync closed remaining Phase 16/17 issue set).

1. Auto-patch tool from raw Dependabot alerts (PR-L1).
- Files: `scripts/dependabot_autopatch_requirements.py`
- Files: `scripts/tests/test_dependabot_autopatch_requirements.py`

2. CI dry-run autopatch visibility and artifacting (PR-L2).
- Files: `.github/workflows/security-audit.yml`
- Files: `.github/workflows/README.md`

3. High/critical patch execution cycle and verification (PR-L3).
- Files: `requirements/eidos_venv_reqs.txt`
- Files: `requirements/eidosian_venv_reqs.txt`
- Files: `doc_forge/requirements.txt`
- Files: `doc_forge/docs/requirements.txt`

4. All-severity deterministic patch pass (PR-L4).
- Files: `requirements/eidos_venv_reqs.txt`
- Files: `requirements/eidosian_venv_reqs.txt`
- Files: `doc_forge/requirements.txt`
- Files: `doc_forge/docs/requirements.txt`

5. No-fix advisory mitigation set (PR-L5).
- Files: `requirements/eidos_venv_reqs.txt`
- Files: `requirements.txt`
- Files: `doc_forge/requirements.txt`
- Files: `doc_forge/docs/requirements.txt`

6. Closure loop (PR-L6).
- Files: `scripts/dependabot_alert_inventory.py`
- Files: `scripts/dependabot_remediation_plan.py`
- Files: `scripts/sync_security_remediation_issues.py`
- Output: zero open alerts + remediation issue closure sync (`closed=4`).

## Stage M (Completed): Runtime Hardening Upgrades

Status checkpoint:
- PR-M1 delivered (kernel module watchdog with consecutive-error quarantine/recovery eventing and persisted watchdog state).
- PR-M2 delivered (event/broadcast payload safety limits with bounded sanitization and truncation telemetry).
- PR-M3 delivered (hardening regression tests and consciousness suite validation).
- PR-M4 delivered (watchdog/payload-safety status surfaced in `eidctl` + MCP runtime status resources/tools).
- PR-M5 delivered (stress benchmark profile for payload-safety overhead + event-bus pressure + CI trend aggregation).

1. Kernel watchdog reliability envelope (PR-M1).
- Files: `agent_forge/src/agent_forge/consciousness/kernel.py`
- Files: `agent_forge/src/agent_forge/consciousness/types.py`

2. Payload safety bounds and truncation telemetry (PR-M2).
- Files: `agent_forge/src/agent_forge/consciousness/types.py`
- Events: `consciousness.payload_truncated`
- Metrics: `consciousness.payload_truncated.count`

3. Regression validation (PR-M3).
- Files: `agent_forge/tests/test_consciousness_kernel_hardening.py`
- Validation: `PYTHONPATH=... ./eidosian_venv/bin/python -m pytest -q agent_forge/tests/test_consciousness_*.py`

4. Watchdog/health visibility in status surfaces (PR-M4).
- Files: `agent_forge/src/agent_forge/consciousness/kernel.py`
- Files: `agent_forge/src/agent_forge/consciousness/trials.py`
- Files: `agent_forge/src/agent_forge/cli/eidctl.py`
- Files: `eidos_mcp/src/eidos_mcp/routers/consciousness.py`

5. Stress benchmark profile + CI trend wiring (PR-M5).
- Files: `agent_forge/src/agent_forge/consciousness/stress.py`
- Files: `agent_forge/tests/test_consciousness_stress_benchmark.py`
- Files: `scripts/consciousness_benchmark_trend.py`
- Files: `scripts/tests/test_consciousness_benchmark_trend.py`
- Files: `.github/workflows/consciousness-parity.yml`

## Stage N (Completed): Linux Audit Matrix Parity

Status checkpoint:
- PR-N1 delivered (deterministic Linux audit matrix runner covering forge, MCP, and consciousness runtime checks).
- PR-N2 delivered (parity smoke script upgraded to execute audit matrix in quick mode for CI-safe runtime).
- PR-N3 delivered (workflow triggers/artifacts extended; audit matrix unit tests added; Part 08 checklist closure).

1. Linux audit matrix runner (PR-N1).
- Files: `scripts/linux_audit_matrix.py`

2. Linux parity smoke integration (PR-N2).
- Files: `scripts/linux_parity_smoke.sh`

3. CI + test + docs parity closure (PR-N3).
- Files: `.github/workflows/consciousness-parity.yml`
- Files: `scripts/tests/test_linux_audit_matrix.py`
- Files: `docs/consciousness_fcl/part-08-termux-linux-hardening.md`
- Files: `docs/consciousness_fcl/PLAN_TRACKER.md`

## Stage O (Completed): Linux Audit Observability Gates

Status checkpoint:
- PR-O1 delivered (trend aggregation now includes Linux audit report coverage and pass/fail metrics).
- PR-O2 delivered (trend regression tests expanded for Linux audit data and markdown rendering).
- PR-O3 delivered (CI parity workflow now enforces latest Linux audit fail-count gate).

1. Trend aggregation integration (PR-O1).
- Files: `scripts/consciousness_benchmark_trend.py`

2. Regression coverage (PR-O2).
- Files: `scripts/tests/test_consciousness_benchmark_trend.py`

3. CI enforcement gate (PR-O3).
- Files: `.github/workflows/consciousness-parity.yml`

## Stage P (Completed): Event Fabric v2 and Marker-Bounded Trial Capture

Status checkpoint:
- PR-P1 delivered (core bus events now emit `event_id` and `ts_ms` alongside existing correlation linkage).
- PR-P2 delivered (EventIndex and TickContext now expose event-id lookup primitives for strict boundary instrumentation).
- PR-P3 delivered (bench runner now emits `bench.trial_start` / `bench.trial_end` markers and captures `events_window` by marker IDs with fallback).
- PR-P4 delivered (regression tests cover event metadata fields and marker-bounded capture boundaries).

1. Event schema enrichment (PR-P1).
- Files: `agent_forge/src/agent_forge/core/events.py`

2. Event-ID indexing and context lookup (PR-P2).
- Files: `agent_forge/src/agent_forge/consciousness/index.py`
- Files: `agent_forge/src/agent_forge/consciousness/types.py`

3. Marker-bounded trial capture (PR-P3).
- Files: `agent_forge/src/agent_forge/consciousness/bench/trials.py`

4. Regression coverage (PR-P4).
- Files: `agent_forge/tests/test_events_corr.py`
- Files: `agent_forge/tests/test_consciousness_bench_trials.py`

## Stage Q (Completed): Trial Provenance and Replay Manifests

Status checkpoint:
- PR-Q1 delivered (bench reporting helper now resolves best-effort git revision for trial provenance).
- PR-Q2 delivered (trial reports now include provenance digest, capture event-id coverage, seed/corr lineage, and kernel beat snapshot).
- PR-Q3 delivered (trial persistence now exports `module_state_snapshot.json` and `replay_manifest.json` alongside existing artifacts).
- PR-Q4 delivered (bench-trial regression assertions expanded to cover provenance and new artifacts).

1. Bench provenance helper (PR-Q1).
- Files: `agent_forge/src/agent_forge/consciousness/bench/reporting.py`

2. Runtime provenance enrichment (PR-Q2).
- Files: `agent_forge/src/agent_forge/consciousness/bench/trials.py`

3. Replay artifact persistence (PR-Q3).
- Files: `agent_forge/src/agent_forge/consciousness/bench/trials.py`

4. Regression validation (PR-Q4).
- Files: `agent_forge/tests/test_consciousness_bench_trials.py`

## Stage R (Completed): Repository Docs Atlas and Validation Cycle

Status checkpoint:
- PR-R1 delivered (automated directory atlas/index generator with deterministic markdown + full index outputs).
- PR-R2 delivered (root/docs/script entrypoints refreshed for modern navigation and explicit directory coverage links).
- PR-R3 delivered (validation cycle executed across consciousness, scripts, MCP tool-call regressions, benchmark and stress benchmark commands).

1. Directory documentation generator (PR-R1).
- Files: `scripts/generate_directory_atlas.py`
- Files: `docs/DIRECTORY_ATLAS.md`
- Files: `docs/DIRECTORY_INDEX_FULL.txt`
- Files: `scripts/tests/test_generate_directory_atlas.py`

2. Entrypoint documentation refresh (PR-R2).
- Files: `README.md`
- Files: `docs/README.md`
- Files: `scripts/README.md`

3. Validation execution (PR-R3).
- Commands: consciousness benchmark + stress benchmark (`eidctl`) and full regression suite in `eidosian_venv`.

## Stage S (Completed): Deterministic Atlas Scope Controls and CI Drift Guard

Status checkpoint:
- PR-S1 delivered (atlas generator now supports deterministic timestamp-free output with optional `--generated-at` plus `tracked|filesystem` scope controls).
- PR-S2 delivered (atlas generator tests expanded for tracked scope rendering, hidden-directory controls, generated-at resolution, and deterministic index output).
- PR-S3 delivered (new CI workflow enforces atlas/index drift gate by regenerating tracked-scope artifacts and failing on diff).
- PR-S4 delivered (docs/trackers refreshed and validation cycle executed with benchmark + stress evidence in `state/bench_phase24`).

1. Deterministic generation and scope controls (PR-S1).
- Files: `scripts/generate_directory_atlas.py`

2. Regression tests for atlas controls (PR-S2).
- Files: `scripts/tests/test_generate_directory_atlas.py`

3. CI drift enforcement workflow (PR-S3).
- Files: `.github/workflows/directory-atlas-drift.yml`
- Files: `.github/workflows/README.md`

4. Docs and validation closure (PR-S4).
- Files: `README.md`
- Files: `docs/README.md`
- Files: `docs/consciousness_fcl/PLAN_TRACKER.md`
- Files: `docs/consciousness_fcl/NEXT_LAYER_PLAN.md`
- Files: `docs/consciousness_fcl/part-47-phase24-deterministic-atlas-drift-guard.md`
- Commands: regression suites + `eidctl consciousness benchmark` + `eidctl consciousness stress-benchmark`.
