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

## Stage E (Active): Continuous Benchmark Expansion

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

## Stage H (Active): Causal Traceability and Consciousness-Lab Instrumentation

Status checkpoint:
- PR-H1 delivered (canonical links + `EventIndex` + `TickContext` index/link helpers + indexed `workspace_competition`/`report` lookups + regression tests).
- PR-H2 delivered (winner-linked `trace_strength` instrumentation + ignition v3 gates + pending winner finalization + regressions).

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
