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

## Validation Commands

1. Agent Forge targeted tests:
```sh
PYTHONPATH=/data/data/com.termux/files/home/eidosian_forge/lib:/data/data/com.termux/files/home/eidosian_forge/agent_forge/src ../eidosian_venv/bin/python -m pytest -q tests/test_consciousness_milestone_a.py tests/test_workspace.py tests/test_db_and_daemon.py
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
