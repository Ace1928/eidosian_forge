# Code Forge Cycle 07: Provenance + Contract + Security Hardening

Date: 2026-02-20

## Scope Completed
1. Roundtrip contract hardening
2. Integration policy controls
3. Provenance model for cross-forge linkage
4. Data governance + secret-scan controls

## Implemented

### Roundtrip Contracts
- Added `code-forge validate-roundtrip` command.
- Added `code_forge/src/code_forge/reconstruct/schema.py` validator for:
  - `reconstruction_manifest.json`
  - `parity_report.json`
  - `roundtrip_summary.json`
  - optional backup `apply_report.json`
- Added hash verification mode (`--verify-hashes`).

### Apply Safety
- `apply_reconstruction(...)` now supports:
  - `require_manifest` guard
  - `dry_run` execution plan mode
- CLI flags added:
  - `apply-reconstruction --require-manifest`
  - `apply-reconstruction --dry-run`

### Integration Policy
- Added policy mode to digest/roundtrip:
  - `run`
  - `effective_run`
  - `global`
- Policy + resolved run scope now persisted in digester summary.

### Provenance Model
- New module: `code_forge/src/code_forge/integration/provenance.py`
- Digester and roundtrip now emit `provenance_links.json`.
- Knowledge sync can emit node links (`unit_id -> node_id`, bounded capture).
- GraphRAG export emits `graphrag_export_manifest.json`.
- MCP tool added: `code_forge_provenance`.

### Data Governance + Secret Safety
- `data/` root blanket ignore removed; runtime subpaths remain ignored by explicit patterns.
- Added `.gitleaks.toml`.
- Added CI workflow `.github/workflows/secret-scan.yml`.
- Added gitleaks pre-commit hook.
- Added governance doc: `docs/DATA_GOVERNANCE.md`.

## Notes
- This cycle prioritizes deterministic traceability and safer promotion gates over feature breadth.
- Remaining high-value next step is signed provenance/roundtrip manifests for tamper evidence.
