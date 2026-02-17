# Part 44: Phase 21 Event Fabric and Marker-Bounded Trial Capture

## Objective

Harden event-level traceability and trial-window capture so experimental artifacts are bounded by explicit runtime markers instead of list-slice heuristics.

Targets delivered in this phase:

1. Add first-class event identity metadata to the core event bus.
2. Add event-id indexing primitives in consciousness indexing/context paths.
3. Capture trial windows using explicit start/end markers, with safe fallback behavior.
4. Add regression tests for schema and boundary guarantees.

## Implemented

### 1) Event schema hardening

File: `agent_forge/src/agent_forge/core/events.py`

`events.append(...)` now emits:

- `event_id`: per-event unique ID.
- `ts_ms`: UTC epoch milliseconds.

Existing fields (`ts`, `type`, `data`, `tags`, `corr_id`, `parent_id`) remain unchanged for backward compatibility.

### 2) Event-ID index support

Files:

- `agent_forge/src/agent_forge/consciousness/index.py`
- `agent_forge/src/agent_forge/consciousness/types.py`

`EventIndex` now tracks `by_event_id`, and `TickContext` exposes `event(event_id)` for O(1)-style lookup in beat-local analysis paths.

### 3) Marker-bounded trial capture

File: `agent_forge/src/agent_forge/consciousness/bench/trials.py`

Bench trials now emit:

- `bench.trial_start`
- `bench.trial_end`

Capture logic now:

1. Attempts strict window extraction from `trial_start.event_id` through `trial_end.event_id`.
2. Falls back to pre-count slicing if marker extraction is unavailable.

Trial reports now include boundary metadata:

- `capture_method`
- `capture_start_event_id`
- `capture_end_event_id`

### 4) Regression coverage

Files:

- `agent_forge/tests/test_events_corr.py`
- `agent_forge/tests/test_consciousness_bench_trials.py`

Coverage added for:

- Presence and typing of `event_id` / `ts_ms` in core event records.
- Existence of `bench.trial_start` and `bench.trial_end` events.
- Marker alignment between `report.json` capture IDs and `events_window.jsonl` boundaries.

## Validation

```sh
PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src ./eidosian_venv/bin/python -m pytest -q \
  agent_forge/tests/test_events_corr.py \
  agent_forge/tests/test_consciousness_bench_trials.py

PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src ./eidosian_venv/bin/python -m pytest -q \
  agent_forge/tests/test_consciousness_*.py \
  agent_forge/tests/test_events_corr.py \
  scripts/tests/test_consciousness_benchmark_trend.py \
  scripts/tests/test_linux_audit_matrix.py

PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src ./eidosian_venv/bin/python -m pytest -q \
  eidos_mcp/tests/test_mcp_tool_calls_individual.py
```

## External References

- Python `uuid` module docs:
- https://docs.python.org/3/library/uuid.html
- JSON Lines format conventions:
- https://jsonlines.org/
