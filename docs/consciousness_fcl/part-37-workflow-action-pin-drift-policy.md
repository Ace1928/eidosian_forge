# Part 37: Workflow Action Pin Drift Policy

## Purpose

Enforce deterministic GitHub Actions dependency references by auditing workflow action refs against a committed lock file.

## Implementation

1. Audit script
- File: `scripts/audit_workflow_action_pins.py`
- Capabilities:
  - scans workflow `uses:` entries
  - resolves action refs to commit SHAs via GitHub API
  - compares resolved SHAs to lock file pins
  - detects mutable refs (e.g. `@main`, `@master`, `@HEAD`)
  - emits structured JSON report and policy exit codes
  - supports lock refresh (`--update-lock`)

2. Lock file
- File: `audit_data/workflow_action_lock.json`
- Contains canonical mapping of action ref keys to resolved SHAs and metadata.

3. CI policy workflow
- File: `.github/workflows/workflow-action-pin-audit.yml`
- Enforced mode:
  - `--enforce-lock`
  - `--fail-on-mutable`
- Uploads audit artifacts for traceability.

4. Script regression tests
- File: `scripts/tests/test_audit_workflow_action_pins.py`
- Covers scan filtering (local/docker ignore) and drift/mutable detection behavior.

## Operational Effect

- Changes to workflow action refs must be accompanied by intentional lock updates.
- Upstream retargeting of mutable refs and lock drift are surfaced as CI failures.
- Action dependency state becomes explicit and reviewable in-repo.
