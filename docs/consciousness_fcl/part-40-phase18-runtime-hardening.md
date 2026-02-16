# Part 40: Phase 18 Runtime Hardening

## Objective

Start a new upgrade cycle after full completion of Phases 0-17 by hardening the consciousness runtime for resilience, safety, and scalability under noisy or adversarial operating conditions.

## Scope

1. Kernel watchdog reliability controls.
2. Payload safety bounds for event and broadcast paths.
3. Regression coverage proving no functional regressions in the consciousness stack.

## Implemented Upgrades

### 1) Kernel watchdog quarantine/recovery

File: `agent_forge/src/agent_forge/consciousness/kernel.py`

- Added persisted watchdog state namespace (`__kernel_watchdog__`) in module state store.
- Added consecutive error tracking per module.
- Added automatic module quarantine when `consecutive_errors >= kernel_watchdog_max_consecutive_errors`.
- Added automatic recovery once cooldown expires.
- Added new events:
- `consciousness.module_quarantined`
- `consciousness.module_recovered`
- Upgraded `consciousness.module_error` data payload to include:
- `consecutive_errors`
- `total_errors`
- `quarantined_until_beat`

Default controls (in config):

- `kernel_watchdog_enabled: true`
- `kernel_watchdog_max_consecutive_errors: 3`
- `kernel_watchdog_quarantine_beats: 20`

### 2) Payload safety envelope

File: `agent_forge/src/agent_forge/consciousness/types.py`

- Added bounded payload sanitizer for all `TickContext.emit_event` and `TickContext.broadcast` writes.
- Added recursive coercion/truncation protection:
- max depth
- max collection items
- max string length
- bounded JSON payload byte size with progressive tightening
- circular reference detection and safe fallback
- Added truncation telemetry event:
- `consciousness.payload_truncated`
- Added metric sample:
- `consciousness.payload_truncated.count`
- Added default safety limits:
- `consciousness_max_payload_bytes: 16384`
- `consciousness_max_depth: 8`
- `consciousness_max_collection_items: 64`
- `consciousness_max_string_chars: 2048`
- `consciousness_payload_truncation_event: true`

## Tests

New test file:

- `agent_forge/tests/test_consciousness_kernel_hardening.py`

Coverage:

1. Watchdog behavior
- repeated failures trigger quarantine
- quarantine period skips module execution
- module recovers automatically after cooldown

2. Payload safety behavior
- large event payloads are truncated safely
- large broadcast payloads are truncated safely
- truncation telemetry events are emitted for both event and broadcast paths

Regression run:

```sh
PYTHONPATH=/data/data/com.termux/files/home/eidosian_forge/lib:/data/data/com.termux/files/home/eidosian_forge/agent_forge/src:/data/data/com.termux/files/home/eidosian_forge/eidos_mcp/src:/data/data/com.termux/files/home/eidosian_forge \
  ./eidosian_venv/bin/python -m pytest -q agent_forge/tests/test_consciousness_*.py
```

Result: `73 passed`.

## Design Notes

- Watchdog state is persisted to maintain deterministic behavior across daemon restarts.
- Quarantine is beat-based (runtime clock independent), matching kernel scheduling semantics.
- Payload safety is applied at write boundaries only, preserving module internals while protecting bus/workspace durability.
- Truncation telemetry is explicit and queryable for observability and benchmark-driven tuning.

## Next Hardening Increments

1. Expose watchdog state and quarantine/recovery counters through `eidctl consciousness status` and MCP status resources.
2. Add payload-safety stress benchmark profile and non-regression gate thresholds to CI trend reporting.
