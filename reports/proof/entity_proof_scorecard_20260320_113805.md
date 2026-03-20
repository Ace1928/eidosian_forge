# Eidos Entity Proof Scorecard

- Generated: `2026-03-20T11:38:05Z`
- Repo root: `/data/data/com.termux/files/home/eidosian_forge`
- Git head: `3c7b4e1201ba5fe959647f178b9b7a01ee7d2a48`
- Worktree dirty: `None`
- Overall status: `yellow`
- Overall score: `0.78625`

## Categories

| Category | Status | Score |
| --- | --- | ---: |
| external_validity | green | 0.95 |
| identity_continuity | green | 1.0 |
| governed_self_modification | green | 0.9 |
| observability | green | 1.0 |
| operational_reproducibility | yellow | 0.7175 |
| adversarial_robustness | red | 0.15 |

## Top Gaps

- `external_validity`: Evidence freshness degraded: 0 stale and 1 missing artifacts within a 30-day window.
- `identity_continuity`: Phenomenological continuity remains weak or zero in the latest surfaced metrics.
- `governed_self_modification`: Change classes, staged deployment, and constitutional approval thresholds are still incomplete.
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

- `scheduler_state`: `sleeping`
- `scheduler_task`: `living_pipeline`
- `scheduler_phase`: `None`
- `scheduler_history_present`: `True`
- `doc_processor_status`: `starting`
- `doc_processor_phase`: `None`
- `local_agent_status`: `timeout`
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
- `overall_delta`: `0.0`

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

## Session Bridge

- `last_sync_at`: `2026-03-20T09:28:11.416901+00:00`
- `recent_sessions`: `6`
- `imported_records`: `43`
- `codex_records`: `32`
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
| 2026-03-20T09:22:12Z | yellow | 0.774583 | yellow | stable |
| 2026-03-20T09:31:32Z | yellow | 0.78125 | yellow | stable |
| 2026-03-20T09:45:26Z | yellow | 0.78625 | yellow | stable |
| 2026-03-20T10:03:20Z | yellow | 0.78625 | yellow | stable |
| 2026-03-20T10:15:28Z | yellow | 0.78625 | yellow | stable |
| 2026-03-20T11:32:19Z | yellow | 0.78625 | yellow | stable |

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
