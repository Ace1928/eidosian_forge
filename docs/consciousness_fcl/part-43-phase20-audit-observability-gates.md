# Part 43: Phase 20 Audit Observability and CI Gates

## Objective

Promote Linux audit matrix output from execution-only telemetry to continuous observability and enforcement:

1. Aggregate Linux audit health into benchmark trends.
2. Track Linux audit pass/fail behavior over time.
3. Enforce latest Linux audit fail-count gate in CI.

## Implemented

### 1) Trend aggregation

File: `scripts/consciousness_benchmark_trend.py`

Added Linux audit ingestion:

- input source: `reports/linux_audit_*.json`
- trend coverage count: `counts.linux_audits`
- metrics:
- `linux_audit.pass_rate`
- `linux_audit.mean_fail_count`
- `linux_audit.mean_checks_total`
- `linux_audit.latest_fail_count`
- `linux_audit.latest_checks_total`
- `linux_audit.latest_id`
- `linux_audit.latest_report`

Markdown rendering now includes Linux audit metrics and latest run ID.

### 2) Regression tests

File: `scripts/tests/test_consciousness_benchmark_trend.py`

Expanded fixtures/assertions to verify:

- Linux audit rows are counted.
- Linux audit pass rate and latest run ID are reported.
- Markdown output includes Linux audit metrics.

### 3) CI gate enforcement

File: `.github/workflows/consciousness-parity.yml`

Added gate step after trend generation:

- fails workflow when `linux_audit.latest_fail_count > 0`
- requires Linux audit metrics to be present in trend report

This ensures parity regressions fail fast instead of silently accumulating.

## Validation Commands

```sh
PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src ./eidosian_venv/bin/python -m pytest -q \
  scripts/tests/test_consciousness_benchmark_trend.py \
  scripts/tests/test_linux_audit_matrix.py

PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src ./eidosian_venv/bin/python \
  scripts/consciousness_benchmark_trend.py --reports-root reports --window-days 30
```

## External References

- GitHub Actions workflow syntax:
- https://docs.github.com/actions/reference/workflows-and-actions/workflow-syntax
- GitHub Actions artifacts and workflow data:
- https://docs.github.com/actions/using-jobs/storing-workflow-data-as-artifacts
