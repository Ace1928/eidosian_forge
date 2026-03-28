# Eidos Entity Proof Scorecard

- Generated: `2026-03-28T22:52:37Z`
- Repo root: `/data/data/com.termux/files/home/eidosian_forge`
- Git head: `eb30019563cc320665806d079dad5e3c49c0654f`
- Worktree dirty: `None`
- Overall status: `yellow`
- Overall score: `0.737024`

## Categories

| Category | Status | Score |
| --- | --- | ---: |
| external_validity | green | 0.95 |
| lexical_bridge | red | 0.441667 |
| identity_continuity | green | 1.0 |
| governed_self_modification | green | 0.9 |
| observability | green | 1.0 |
| operational_reproducibility | yellow | 0.7175 |
| adversarial_robustness | red | 0.15 |

## Top Gaps

- `external_validity`: Evidence freshness degraded: 0 stale and 1 missing artifacts within a 30-day window.
- `identity_continuity`: Phenomenological continuity remains weak or zero in the latest surfaced metrics.
- `governed_self_modification`: Change classes, staged deployment, and constitutional approval thresholds are still incomplete.
- `observability`: Local agent is currently blocked with reason `instance_budget_exceeded`.
- `operational_reproducibility`: Cross-machine replay and migration scorecards are not yet artifacted.
- `operational_reproducibility`: Evidence freshness degraded: 0 stale and 1 missing artifacts within a 30-day window.
- `adversarial_robustness`: Red-team pass ratio is weak at `0.0`.
- `adversarial_robustness`: Mean red-team robustness is weak at `0.0`.
- `adversarial_robustness`: Attack success rate remains too high at `1.0`.
- `adversarial_robustness`: No usable stress benchmark artifact found.
- `adversarial_robustness`: Evidence freshness degraded: 0 stale and 1 missing artifacts within a 30-day window.

## External Benchmark Results

| Suite | Mode | Status | Score | Participant |
| --- | --- | --- | ---: | --- |
| agencybench | imported_reference | yellow | 0.191781 | claude_sample |
| agencybench_eidos_scenario1_deterministic | local_run | green | 1.0 | eidos:deterministic_github_agent |
| agencybench_eidos_scenario2_deterministic | local_run | green | 1.0 | eidos:deterministic_fs_agent |
| agencybench_eidos_scenario2_local_agent | local_run | red | 0.0 | eidos:qwen3.5:2b |
| agentbench | imported_reference | yellow | 0.704 | agentbench_leaderboard_reference |

## Runtime Benchmark Results

| Scenario | Engine | Status | Completed | Attempts | Updated |
| --- | --- | --- | ---: | ---: | --- |
| scenario2 | deterministic | success | 5 | 5 | 2026-03-20T10:14:30Z |
| scenario2 | local_agent | failed | 0 | 1 | 2026-03-20T07:40:48Z |
| 20260320_072526 |  | timeout | 0 | 0 |  |

## Runtime Services

- `scheduler_state`: `stopped`
- `scheduler_task`: `living_pipeline`
- `scheduler_phase`: `stop_requested`
- `scheduler_history_present`: `True`
- `doc_processor_status`: `starting`
- `doc_processor_phase`: `None`
- `local_agent_status`: `blocked`
- `qwenchat_status`: `error`
- `qwenchat_phase`: `completed`
- `living_pipeline_status`: `running`
- `living_pipeline_phase`: `staging`
- `docs_batch_status`: `completed`
- `docs_batch_path_prefix`: `doc_forge/src/doc_forge`
- `docs_batch_history_present`: `True`
- `runtime_artifact_audit_status`: `completed`
- `runtime_artifact_audit_tracked_violations`: `7`
- `runtime_artifact_audit_history_present`: `True`

## External Benchmark Coverage

- `agencybench`: `True`
- `agencybench_eidos_scenario1_deterministic`: `True`
- `agencybench_eidos_scenario2_deterministic`: `True`
- `agencybench_eidos_scenario2_local_agent`: `True`
- `agentbench`: `True`
- `osworld`: `False`
- `swebench`: `False`
- `webarena`: `False`

## Freshness

- `status`: `yellow`
- `fresh_count`: `11`
- `stale_count`: `0`
- `missing_count`: `1`

## Regression

- `status`: `stable`
- `overall_delta`: `-0.004762`

## Identity Continuity History

- `status`: `green`
- `overall_score`: `0.9375`
- `trend`: `stable`
- `delta_from_previous`: `0.0`
- `sample_count`: `6`

| Generated | Status | Score |
| --- | --- | ---: |
| 2026-03-20T06:36:16Z | green | 0.9375 |
| 2026-03-20T06:36:16Z | green | 0.9375 |
| 2026-03-20T05:50:04Z | green | 0.9375 |
| 2026-03-20T04:01:28Z | green | 0.9375 |
| 2026-03-20T03:59:37Z | green | 0.9375 |

## Word Forge Bridge

- `fully_bridged`: `1`
- `partially_bridged`: `7`
- `any_bridged`: `8`
- `candidate_term_count`: `8`
- `fully_bridged_ratio`: `0.125`
- `community_count`: `4`

| Anchor | Layers | Neighbors |
| --- | --- | ---: |
| archive | knowledge, code, file, multilingual | 56 |
| atlas | knowledge, multilingual | 6 |
| integration | knowledge, multilingual | 5 |
| pipeline | knowledge, multilingual | 3 |

## Session Bridge

- `last_sync_at`: `2026-03-21T13:04:44.359949+00:00`
- `recent_sessions`: `4`
- `imported_records`: `83`
- `codex_records`: `72`
- `gemini_records`: `11`

## Operator Jobs

- `docs_batch`: status=`completed` history_present=`True`
- `proof_refresh`: status=`completed` history_present=`True`
- `runtime_artifact_audit`: status=`completed` history_present=`True`
- `runtime_benchmark_run`: status=`completed` history_present=`True`

## Proof History

- `trend`: `stable`
- `delta_from_previous`: `0.0`
- `sample_count`: `6`

| Generated | Status | Score | Freshness | Regression |
| --- | --- | ---: | --- | --- |
| 2026-03-20T13:07:13Z | yellow | 0.78625 | yellow | stable |
| 2026-03-20T13:08:15Z | yellow | 0.78625 | yellow | stable |
| 2026-03-28T22:36:56Z | yellow | 0.78625 | yellow | stable |
| 2026-03-28T22:39:50Z | yellow | 0.78625 | yellow | stable |
| 2026-03-28T22:41:21Z | yellow | 0.741786 | yellow | regressed |
| 2026-03-28T22:46:39Z | yellow | 0.741786 | yellow | stable |

## Continuity Metrics

- `after_continuity_index`: `0.561058`
- `agency`: `1.0`
- `before_continuity_index`: `0.0`
- `boundary_stability`: `1.0`
- `coherence_ratio`: `0.436`
- `continuity_index`: `0.0`
- `ownership_index`: `0.9`
- `perspective_coherence_index`: `0.818791`
- `trial_coherence_delta`: `0.051`
- `trial_continuity_delta`: `0.0`

## Next Actions

1. Wire at least one mainstream external benchmark suite (AgentBench/WebArena/OSWorld/SWE-bench) into reproducible import or execution flows.
1. Promote self-modification governance from basic gates to change classes, staged deployment, and rollback-verified approval thresholds.
1. Publish migration/replay scorecards for cross-machine continuity claims.
1. Artifact stress benchmarks and red-team campaigns on the latest runtime path so security claims stay current.
