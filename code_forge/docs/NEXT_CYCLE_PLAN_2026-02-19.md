# Code Forge Next Cycle Plan (Post-Roundtrip Foundation)

Date: 2026-02-19  
Scope: Scale the roundtrip workflow from one validated forge (`audit_forge`) to broader forge coverage with stronger contracts and performance guarantees.

## Phase 1: Contract Hardening
- [ ] Add signed manifest option (`reconstruction_manifest.sig`) for tamper evidence.
- [ ] Add schema validator for roundtrip artifacts (`reconstruction_manifest`, `parity_report`, `roundtrip_summary`, `apply_report`).
- [ ] Add CLI `validate-roundtrip` for contract checks across an artifact directory.
- [ ] Add deterministic ordering assertion for reconstruction manifests across reruns.

## Phase 2: Performance and Scale
- [ ] Add parallel hashing mode for parity checks on large trees.
- [ ] Add benchmark scenarios for roundtrip (small/medium/large forges).
- [ ] Add regression gates for:
  - parity runtime,
  - reconstruction runtime,
  - apply runtime,
  - memory footprint.
- [ ] Add explicit `--jobs` control for reconstruction/parity.

## Phase 3: Multi-Forge Rollout
- [ ] Run roundtrip validation for:
  - `diagnostics_forge`
  - `erais_forge`
  - `article_forge`
- [ ] Capture per-forge artifact packs and parity evidence under `data/code_forge/roundtrip/<forge>/`.
- [ ] Track coverage matrix: parity pass rate, apply mode outcome, integration export stats.

## Phase 4: Integration Elevation
- [ ] Add root-scoped export policy mode flag:
  - `run` (active run),
  - `effective_run` (fallback),
  - `global` (legacy/global).
- [ ] Add Code Forge -> Knowledge Forge relationship typing for roundtrip artifacts.
- [ ] Add Code Forge -> GraphRAG export manifest with run/provenance metadata.

## Phase 5: Governance and Promotion
- [ ] Add promotion gate:
  - require parity pass,
  - require benchmark pass,
  - require roundtrip contract validation pass.
- [ ] Add optional auto-generated migration/promotion report for human signoff.
- [ ] Add CI workflow job for roundtrip smoke test on a fixture forge.
