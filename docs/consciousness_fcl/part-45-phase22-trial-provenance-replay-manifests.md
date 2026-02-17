# Part 45: Phase 22 Trial Provenance and Replay Manifests

## Objective

Harden bench-trial artifacts so every trial result includes replay-grade provenance:

1. Runtime/commit identity context.
2. Deterministic capture digest and boundary metadata.
3. Persisted module-state snapshot for post-run reconstruction.
4. Dedicated replay manifest for tooling and CI pipelines.

## Implemented

### 1) Provenance helper

File: `agent_forge/src/agent_forge/consciousness/bench/reporting.py`

Added best-effort Git revision resolver:

- `git_revision(path: Path) -> str | None`

This keeps provenance collection non-fatal in non-repo contexts (tmp-state tests, isolated runners).

### 2) Trial report provenance enrichment

File: `agent_forge/src/agent_forge/consciousness/bench/trials.py`

Each trial report now includes `provenance` with:

- `git_sha`
- `seed`
- `trial_corr_id`
- `capture_event_digest`
- `capture_event_id_coverage`
- `capture_start_ts`
- `capture_end_ts`
- `module_state_count`
- `kernel_beat_count`

Digest is computed from ordered captured event IDs (or stable fallback keys), enabling integrity checks across exports.

### 3) Replay-grade artifacts

File: `agent_forge/src/agent_forge/consciousness/bench/trials.py`

Persisted trial artifact set now includes:

- `module_state_snapshot.json`
- `replay_manifest.json`

`replay_manifest.json` records boundary IDs, digest, seed, and git SHA for deterministic replay tooling.

### 4) Test coverage

File: `agent_forge/tests/test_consciousness_bench_trials.py`

Assertions now verify:

- New artifact files exist.
- Report includes provenance keys.
- Existing marker-boundary capture guarantees remain valid.

## Validation

```sh
PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src ./eidosian_venv/bin/python -m pytest -q \
  agent_forge/tests/test_consciousness_bench_trials.py \
  agent_forge/tests/test_events_corr.py

PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src ./eidosian_venv/bin/python -m pytest -q \
  agent_forge/tests/test_consciousness_*.py \
  agent_forge/tests/test_events_corr.py \
  scripts/tests/test_consciousness_benchmark_trend.py \
  scripts/tests/test_linux_audit_matrix.py

PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src ./eidosian_venv/bin/python -m pytest -q \
  eidos_mcp/tests/test_mcp_tool_calls_individual.py
```

## External References

- Git commit identity and revision plumbing:
- https://git-scm.com/docs/git-rev-parse
- Python `hashlib` reference:
- https://docs.python.org/3/library/hashlib.html
