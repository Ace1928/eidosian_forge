# Part 34: Benchmark Index Migration

## Purpose

Complete the causal-instrumentation migration by moving benchmark extraction paths off repeated full-window scans and onto `EventIndex` lookups.

## Scope

- Extend `EventIndex` with per-type event buckets.
- Use indexed lookups for benchmark snapshot metric extraction.
- Use indexed lookups for trial report counters and gate inputs.

## Implementation

1. Event index extension
- File: `agent_forge/src/agent_forge/consciousness/index.py`
- Added `EventIndex.by_type: dict[str, list[dict]]`.
- `build_index(...)` now populates `by_type` alongside `latest_by_type`, broadcast kind buckets, and linkage maps.

2. Bench snapshot extraction migration
- File: `agent_forge/src/agent_forge/consciousness/bench/trials.py`
- `_snapshot(...)` now builds one `EventIndex` for `recent_events` and reads:
  - latest `phenom.snapshot` via `index.latest_by_type`
  - latest metric samples via `index.by_type["metrics.sample"]`

3. Trial window metric/counter migration
- File: `agent_forge/src/agent_forge/consciousness/bench/trials.py`
- `run_trial(...)` now computes:
  - `event_type_counts` from `index.by_type`
  - `module_error_count` from `index.by_type["consciousness.module_error"]`
  - `degraded_mode_ratio` from `index.by_type["meta.state_estimate"]`
  - `winner_count` from `index.broadcasts_by_kind["GW_WINNER"]`
  - `ignitions_without_trace` from `index.by_type["gw.ignite"]`

## Why This Matters

- Keeps benchmark extraction aligned with the canonical indexing contract already used by runtime modules.
- Reduces repeated full-list scans and duplicated event-type branching in trial scoring.
- Improves determinism and maintainability for future benchmark expansion.

## Validation

```sh
PYTHONPATH=lib:agent_forge/src:memory_forge/src:knowledge_forge/src:eidos_mcp/src ./eidosian_venv/bin/python -m pytest -q agent_forge/tests/test_consciousness_linking_index.py
PYTHONPATH=lib:agent_forge/src:memory_forge/src:knowledge_forge/src:eidos_mcp/src ./eidosian_venv/bin/python -m pytest -q agent_forge/tests/test_consciousness_bench_trials.py
PYTHONPATH=lib:agent_forge/src:memory_forge/src:knowledge_forge/src:eidos_mcp/src ./eidosian_venv/bin/python -m pytest -q agent_forge/tests/test_consciousness_*.py
```
