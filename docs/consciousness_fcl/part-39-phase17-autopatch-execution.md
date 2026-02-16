# Part 39: Phase 17 Auto-Patch Execution Loop

## Objective

Convert Phase 16 planning outputs into production remediation execution by generating deterministic requirement pin updates from Dependabot raw alerts and validating idempotent, rollback-safe application in Termux/Linux.

## Scope

- New tool: `scripts/dependabot_autopatch_requirements.py`
- Inputs: raw Dependabot alert JSON (`gh api repos/<owner>/<repo>/dependabot/alerts --paginate`)
- Output: dry-run/write reports (`JSON` + `Markdown`) and optional file backups
- Target: pip requirement manifests with pinned entries (`name==version`)

## Implementation

1. Alert ingestion and filtering
- Reads raw alerts and filters by `state`, `severity`, and `ecosystem`.
- Extracts `dependency.manifest_path`, package name, and `security_vulnerability.first_patched_version.identifier`.

2. Deterministic target resolution
- Groups by `manifest_path + package`.
- Selects highest required patched version.
- Preserves strongest severity and merged alert-number evidence.

3. Safe patching semantics
- Updates only pinned lines (`==`) and preserves extras, markers, and comments.
- Flags unresolved packages as:
- `unresolved_missing` when no matching requirement is present.
- `unresolved_unpinned` when present but not pinned (`>=`, `~=`, direct refs, etc.).
- Defaults to `dry_run`; write mode requires `--write`.

4. Rollback and auditability
- Write mode stores timestamped backups under `Backups/security_dependency_patches/<UTC-stamp>/`.
- Emits structured report with per-manifest change set, unresolved counts, and evidence metadata.

## Validation Results (Current Cycle)

- High/Critical dry-run (`open`, `pip`) reported:
- `manifests=4`, `changed=4`, `updated=45`, `unresolved_missing=0`, `unresolved_unpinned=0`.
- Write pass applied those 45 updates across:
- `requirements/eidos_venv_reqs.txt`
- `requirements/eidosian_venv_reqs.txt`
- `doc_forge/requirements.txt`
- `doc_forge/docs/requirements.txt`
- Post-write dry-run confirmed idempotency:
- `changed=0`, `updated=0`, `unresolved=0/0`.

## Iteration 2 (All-Severity + No-Fix Mitigations)

1. Auto-patch expansion
- Ran all-severity (`critical,high,medium,low`) write pass.
- Applied additional `32` deterministic pin updates.
- Post-write dry-run remained idempotent (`changed=0`, `updated=0`).

2. No-fix advisory handling
- Advisories without `first_patched_version` were reduced by dependency minimization and manual range-safe bumping:
- Removed direct `ecdsa` dependency from:
- `requirements.txt`
- `requirements/eidos_venv_reqs.txt`
- Removed direct `keras` dependency from:
- `doc_forge/requirements.txt`
- `doc_forge/docs/requirements.txt`
- Manually bumped `orjson` in `requirements/eidos_venv_reqs.txt`:
- `3.10.16 -> 3.11.5` (above vulnerable range `<= 3.11.4`).

3. Local closure estimate
- Using raw alert metadata + current manifest scan, local estimator reports `0` remaining open vulnerable requirements mappings.
- GitHub Dependabot UI/API may continue reporting stale totals until repository re-analysis completes.

## CI Integration

`/.github/workflows/security-audit.yml` now also:

- Exports raw Dependabot alerts artifact (`dependabot_alerts_<state>_raw.json`).
- Runs auto-patch tool in dry-run mode and uploads reports.
- Publishes autopatch summary counters in workflow summary.

## Operational Commands

```sh
# Dry-run plan only (safe default)
./eidosian_venv/bin/python scripts/dependabot_autopatch_requirements.py \
  --alerts-json reports/security/dependabot_alerts_raw.jsonl \
  --repo-root . \
  --severity critical,high \
  --state open \
  --ecosystem pip \
  --json-out reports/security/dependabot_autopatch_highcrit.dryrun.json \
  --md-out reports/security/dependabot_autopatch_highcrit.dryrun.md

# Apply updates with rollback backup
./eidosian_venv/bin/python scripts/dependabot_autopatch_requirements.py \
  --alerts-json reports/security/dependabot_alerts_raw.jsonl \
  --repo-root . \
  --severity critical,high \
  --state open \
  --ecosystem pip \
  --write \
  --backup-root Backups/security_dependency_patches \
  --json-out reports/security/dependabot_autopatch_highcrit.write.json \
  --md-out reports/security/dependabot_autopatch_highcrit.write.md
```

## Sources

- GitHub REST Dependabot Alerts:
  - https://docs.github.com/en/rest/dependabot/alerts?apiVersion=2022-11-28
- pip requirements file format:
  - https://pip.pypa.io/en/stable/reference/requirements-file-format/
