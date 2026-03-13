# Entity Proof Execution Program

Date: 2026-03-13
Status: Active subordinate execution slice
Parent program: `docs/plans/EIDOSIAN_MASTER_IMPLEMENTATION_PROGRAM_2026-03-07.md`

## Intent

Convert Eidos from an internally compelling system into an externally legible, reproducible, benchmarked, and governable system by building a reportable proof layer over:
- capability benchmarks,
- continuity and identity metrics,
- observability/telemetry,
- governed self-modification,
- adversarial robustness,
- reproducibility and migration evidence.

## Gap Framing

The current forge already contains substantial internal evidence:
- consciousness benchmarks and trials,
- RAC-AP validation,
- red-team/autotune guards,
- GraphRAG quality assessment,
- model-domain suites,
- runtime coordinator and local-agent telemetry.

The missing layer is a canonical external-proof contract that:
- aggregates that evidence coherently,
- highlights what is still missing,
- is comparable across runs,
- and can be published or reviewed without hidden context.

## Completed

- [x] Audited the existing benchmark/eval/telemetry surfaces against the external-proof requirements.
- [x] Added canonical proof scorecard generator in `scripts/entity_proof_suite.py`.
- [x] Added reportable contracts and latest artifacts under `reports/proof/`.
- [x] Added regression coverage in `scripts/tests/test_entity_proof_suite.py`.
- [x] Exposed latest proof snapshot through Atlas runtime/API surfaces.
- [x] Generated first live proof artifact:
  - `reports/proof/entity_proof_scorecard_20260313_023705.json`
  - `reports/proof/entity_proof_scorecard_20260313_023705.md`
  - overall status: `yellow`
  - overall score: `0.686667`
- [x] Added freshness policy and stale-evidence degradation to the proof scorecard.
- [x] Added previous-scorecard regression comparison and surfaced deltas in the proof bundle.
- [x] Added external benchmark evidence import contract:
  - `scripts/import_external_benchmark.py`
  - `reports/external_benchmarks/<suite>/latest.json`
- [x] Added bounded AgencyBench reference importer:
  - `scripts/import_agencybench_reference.py`
- [x] Added migration/replay scorecard contract:
  - `scripts/migration_replay_scorecard.py`
  - `reports/proof/migration_replay_scorecard_latest.json`
- [x] Added scheduler-side proof runtime normalization:
  - `data/runtime/entity_proof_status.json`
- [x] Added first real live Eidos-run external benchmark harness:
  - `scripts/run_agencybench_eidos.py`
- [x] Generated live AgencyBench scenario2 deterministic artifact:
  - `reports/external_benchmarks/agencybench_eidos_scenario2_deterministic/latest.json`
  - score: `1.0`
- [x] Generated live AgencyBench scenario1 deterministic artifact:
  - `reports/external_benchmarks/agencybench_eidos_scenario1_deterministic/latest.json`
  - score: `1.0`
- [x] Refreshed proof scorecard after live external Eidos-run artifacts:
  - `reports/proof/entity_proof_scorecard_20260313_072818.json`
  - overall score: `0.703282`
  - status: `yellow`

## Open Work

- [~] Wire at least one mainstream external suite into reproducible import or execution flows:
  - [x] import contract for upstream results
  - [x] bounded official AgencyBench sample-reference import path
  - [x] at least one live imported suite artifact on the mainline runtime
  - [x] optional bounded local execution harness for one suite
  - [x] second live external benchmark artifact beyond the first proof run
  - [ ] non-deterministic live model-driven benchmark path that is stable enough for publication
  - Candidate suites:
    - AgencyBench
    - AgentBench
    - WebArena
    - OSWorld
    - SWE-bench
- [x] Add benchmark freshness and regression gates so stale evidence is explicitly marked as degraded.
- [x] Add replay/migration scorecards for cross-machine continuity claims.
- [x] Add a canonical theory-of-operation document and make it part of the proof bundle.
- [ ] Promote self-modification governance from basic gates to change classes, staged deployment, and rollback-verified approval thresholds.
- [ ] Add explicit identity continuity scorecards across upgrades, not just raw continuity metrics.
- [ ] Add benchmark artifact publication for external review bundles.

## Validation

- [x] `scripts/tests/test_entity_proof_suite.py`
- [x] `web_interface_forge/tests/test_dashboard.py`

## Next Execution Slice

1. stabilize the `qwen3.5:2b` live benchmark execution path so `local_agent` scenario runs can replace deterministic fallback as publication-grade evidence
2. add publishable proof-bundle export with scorecard, migration scorecard, theory of operation, and benchmark manifests
3. add identity continuity scorecards across upgrade boundaries
