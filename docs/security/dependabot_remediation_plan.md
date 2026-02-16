# Dependabot Remediation Plan

Last inventory pull: 2026-02-16 (live via GitHub API)

## Current State

- Open alerts: 165
- Severity split: 8 critical, 62 high, 70 medium, 25 low
- Ecosystems: 179 `pip`, 1 `npm` (historical total)
- Highest alert concentration by manifest path:
1. `requirements/eidos_venv_reqs.txt` (77)
2. `doc_forge/requirements.txt` (38)
3. `doc_forge/docs/requirements.txt` (38)
4. `requirements/eidosian_venv_reqs.txt` (11)

Machine-readable snapshot:
- `docs/security/dependabot_open_summary_2026-02-16.json`

## Triage Strategy

1. Deduplicate root causes by package and version constraints, not by alert count.
2. Patch critical/high first across shared manifests.
3. Rebuild lock/install baselines in Termux and Linux parity modes.
4. Run regression suites before each batch commit.
5. Close alerts only after upstream versions are verified in all relevant manifests.

## Wave Plan

1. Wave 1 (critical/high)
- Target packages (first pass): `unstructured`, `urllib3`, `aiohttp`, `keras`, `protobuf`, `python-multipart`, `wheel`, `cryptography`, `pillow`, `nbconvert`, `pyasn1`
- Scope: all affected manifests in one branch wave with grouped commits by compatibility cluster.

2. Wave 2 (medium)
- Target packages: `requests`, `filelock`, `starlette`, `authlib`, `orjson`, `pypdf`, others in inventory.
- Scope: resolve with minimal API surface changes and strict test gating.

3. Wave 3 (low + hygiene)
- Clean residual low severity items and dependency pin drift.
- Normalize requirement files to reduce repeated alert fan-out.

## Execution Commands

Inventory refresh:
```bash
./scripts/dependabot_alert_inventory.py \
  --repo Ace1928/eidosian_forge \
  --state open \
  --json-out docs/security/dependabot_open_summary_$(date +%F).json
```

## Acceptance Criteria

1. Open critical alerts = 0.
2. Open high alerts = 0.
3. Termux parity smoke passes.
4. Linux parity smoke passes.
5. MCP tool/resource audit shows no new hard failures.
