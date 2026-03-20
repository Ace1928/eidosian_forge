# Eidos Entity Proof Scorecard

- Generated: `2026-03-20T09:31:32Z`
- Repo root: `/data/data/com.termux/files/home/eidosian_forge`
- Git head: `3c7b4e1201ba5fe959647f178b9b7a01ee7d2a48`
- Worktree dirty: `None`
- Overall status: `yellow`
- Overall score: `0.78125`

## Categories

| Category | Status | Score |
| --- | --- | ---: |
| external_validity | green | 0.95 |
| identity_continuity | green | 1.0 |
| governed_self_modification | green | 0.9 |
| observability | green | 1.0 |
| operational_reproducibility | yellow | 0.6875 |
| adversarial_robustness | red | 0.15 |

## Top Gaps

- `external_validity`: Runtime benchmark traces exist, but none currently show a successful completion.
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
| scenario2 | local_agent | failed | 0 | 1 | 2026-03-20T07:40:48Z |
| 20260320_072526 |  | timeout | 0 | 0 |  |

## Runtime Services

- `scheduler_state`: `sleeping`
- `doc_processor_status`: `starting`
- `doc_processor_phase`: `None`
- `local_agent_status`: `timeout`
- `qwenchat_status`: `running`
- `qwenchat_phase`: `model_request`
- `living_pipeline_status`: `running`
- `living_pipeline_phase`: `staging`

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
- `overall_delta`: `0.006667`

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

## Proof History

- `trend`: `stable`
- `delta_from_previous`: `0.02`
- `sample_count`: `6`

| Generated | Status | Score | Freshness | Regression |
| --- | --- | ---: | --- | --- |
| 2026-03-20T06:38:09Z | yellow | 0.739453 | yellow | stable |
| 2026-03-20T06:51:06Z | yellow | 0.764453 | yellow | improved |
| 2026-03-20T07:08:49Z | yellow | 0.747917 | yellow | regressed |
| 2026-03-20T07:37:52Z | yellow | 0.75625 | yellow | stable |
| 2026-03-20T09:12:57Z | yellow | 0.754583 | yellow | stable |
| 2026-03-20T09:22:12Z | yellow | 0.774583 | yellow | stable |

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
