# Part 35: Workflow Security Audit Loop

## Purpose

Add an operational security feedback loop that continuously inventories dependency risk and produces actionable artifacts without blocking normal development by default.

## Scope

1. Scheduled and manual Dependabot inventory workflow.
2. Structured report artifact emission.
3. Optional hard-fail thresholds for critical/high open alerts.

## Implementation

1. New workflow
- File: `.github/workflows/security-audit.yml`
- Runs on:
  - schedule (`daily`)
  - `workflow_dispatch`
- Permissions:
  - `contents: read`
  - `security-events: read`

2. Data collection
- Uses existing script:
  - `scripts/dependabot_alert_inventory.py`
- Outputs JSON report under:
  - `reports/security/dependabot_alerts_<state>.json`
- Captures command logs under:
  - `reports/security/dependabot_alerts_<state>.log`

3. Failure-safe artifacting
- If API access fails (token scope/environment mismatch), workflow writes fallback error JSON and still uploads artifacts for diagnosis.

4. Optional enforcement gates
- Manual inputs:
  - `fail_on_critical`
  - `fail_on_high`
- If enabled, workflow fails when open severity counts exceed zero for selected levels.

## Why This Matters

- Converts repository-level security posture into a repeatable machine-readable signal.
- Supports strict security sweeps on demand while keeping scheduled observability resilient.
- Establishes a baseline loop for later automated remediation phases.

## Validation

```sh
python scripts/dependabot_alert_inventory.py --help
```

`workflow-lint` in CI validates workflow syntax and expression structure.
