# Part 42: Phase 19 Linux Audit Matrix Parity

## Objective

Close Linux parity drift by running a deterministic audit matrix in CI that validates the same runtime control surfaces used in Termux:

1. Forge availability/status.
2. MCP tools/resources hard-fail matrix.
3. Consciousness runtime status health (watchdog + payload safety).
4. Consciousness benchmark and stress benchmark gates.

## Implemented

### 1) Linux audit matrix runner

File: `scripts/linux_audit_matrix.py`

Added a production-safe audit runner with:

- strict JSON evaluation helpers for each subsystem.
- report persistence (`reports/linux_audit_<timestamp>_<id>.json`).
- quick mode for CI runtime control.
- strict mode for optional future gate tightening.

Checks in the matrix:

- `forge_status`
- `mcp_audit_matrix`
- `consciousness_status`
- `consciousness_benchmark`
- `consciousness_stress_benchmark`
- optional `consciousness_full_benchmark` (disabled in quick mode)

### 2) Linux smoke integration

File: `scripts/linux_parity_smoke.sh`

Replaced standalone MCP audit invocation with unified matrix call:

- `python scripts/linux_audit_matrix.py --quick --timeout 240 --mcp-timeout 8 --report-dir reports`

This keeps smoke parity deterministic while ensuring the same matrix is exercised for CI and local Linux runs.

### 3) CI wiring

File: `.github/workflows/consciousness-parity.yml`

Updated:

- path triggers include new script/test files.
- artifacts now include `reports/linux_audit_*.json`.

### 4) Tests

File: `scripts/tests/test_linux_audit_matrix.py`

Added unit tests for:

- report path parsing.
- forge status evaluator.
- MCP strict/soft-fail behavior.
- consciousness status evaluator requirements.
- stress benchmark evaluator gate/truncation checks.

## Part 08 Closure

File: `docs/consciousness_fcl/part-08-termux-linux-hardening.md`

Marked the remaining unchecked checklist item as complete:

- Linux CI runner now validates parity against the same audit matrix.

## Validation Commands

```sh
PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src ./eidosian_venv/bin/python -m pytest -q \
  scripts/tests/test_linux_audit_matrix.py \
  scripts/tests/test_consciousness_benchmark_trend.py \
  agent_forge/tests/test_consciousness_trials.py

./eidosian_venv/bin/python scripts/linux_audit_matrix.py --quick --report-dir reports
```

## External References

- GitHub Actions workflow syntax:
- https://docs.github.com/actions/reference/workflows-and-actions/workflow-syntax
- GitHub Actions artifact guidance:
- https://docs.github.com/actions/using-jobs/storing-workflow-data-as-artifacts
- Pytest exit code contract:
- https://docs.pytest.org/en/stable/reference/exit-codes.html
