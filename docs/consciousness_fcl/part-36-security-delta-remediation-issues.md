# Part 36: Security Delta Remediation Issues

## Purpose

Add automated issue-driven remediation tracking from dependency security inventory deltas.

## Implementation

- Workflow: `.github/workflows/security-audit.yml`
- Added `issues: write` permission.
- Added issue automation step using `actions/github-script@v7`:
  - Reads generated security inventory JSON report.
  - Parses prior baseline from issue body marker.
  - Computes deltas for open total + critical/high/moderate/low.
  - Creates or updates a dedicated issue (`Security Dependency Delta Report`) when risk is active or regressing.
  - Closes existing issue automatically when no high/critical risk remains and no regression is detected.

## Idempotence and Traceability

- Issue body contains stable markers:
  - `<!-- eidos-security-audit -->`
  - `<!-- eidos-security-baseline:{...} -->`
- Each run updates the baseline marker to current measured state.
- Repeated runs with no changes do not create duplicate issues.

## Operational Effect

- Security inventory transitions now create an explicit remediation thread in GitHub Issues.
- The loop is automated and stateful without requiring persistent local state between runs.
