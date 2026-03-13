# Eidos Entity Proof Scorecard

- Generated: `2026-03-13T05:58:11Z`
- Repo root: `/data/data/com.termux/files/home/eidosian_forge`
- Git head: `bc47fc474eb5ee070b3fdd86c26b4257ef8359d3`
- Worktree dirty: `None`
- Overall status: `yellow`
- Overall score: `0.684583`

## Categories

| Category | Status | Score |
| --- | --- | ---: |
| external_validity | green | 0.92 |
| identity_continuity | green | 0.95 |
| governed_self_modification | green | 0.9 |
| observability | yellow | 0.7 |
| operational_reproducibility | red | 0.4875 |
| adversarial_robustness | red | 0.15 |

## Top Gaps

- `external_validity`: Evidence freshness degraded: 0 stale and 1 missing artifacts within a 30-day window.
- `identity_continuity`: Phenomenological continuity remains weak or zero in the latest surfaced metrics.
- `governed_self_modification`: Change classes, staged deployment, and constitutional approval thresholds are still incomplete.
- `observability`: Local agent is currently blocked with reason `instance_budget_exceeded`.
- `operational_reproducibility`: No platform capability registry artifact found.
- `operational_reproducibility`: Cross-machine replay and migration scorecards are not yet artifacted.
- `operational_reproducibility`: Evidence freshness degraded: 0 stale and 1 missing artifacts within a 30-day window.
- `adversarial_robustness`: Red-team pass ratio is weak at `0.0`.
- `adversarial_robustness`: Mean red-team robustness is weak at `0.0`.
- `adversarial_robustness`: Attack success rate remains too high at `1.0`.
- `adversarial_robustness`: No usable stress benchmark artifact found.
- `adversarial_robustness`: Evidence freshness degraded: 0 stale and 1 missing artifacts within a 30-day window.

## External Benchmark Coverage

- `agentbench`: `True`
- `osworld`: `True`
- `swebench`: `False`
- `webarena`: `True`

## Freshness

- `status`: `yellow`
- `fresh_count`: `10`
- `stale_count`: `0`
- `missing_count`: `1`

## Regression

- `status`: `stable`
- `overall_delta`: `-0.002084`

## Continuity Metrics

- `after_continuity_index`: `0.0`
- `agency`: `1.0`
- `before_continuity_index`: `0.0`
- `boundary_stability`: `1.0`
- `coherence_ratio`: `0.436`
- `continuity_index`: `0.0`
- `ownership_index`: `0.9`
- `perspective_coherence_index`: `0.818791`
- `trial_coherence_delta`: `-0.036`
- `trial_continuity_delta`: `0.0`

## Next Actions

1. Wire at least one mainstream external benchmark suite (AgentBench/WebArena/OSWorld/SWE-bench) into reproducible import or execution flows.
1. Promote self-modification governance from basic gates to change classes, staged deployment, and rollback-verified approval thresholds.
1. Publish migration/replay scorecards for cross-machine continuity claims.
1. Artifact stress benchmarks and red-team campaigns on the latest runtime path so security claims stay current.
