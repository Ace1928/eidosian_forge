# Part 41: Phase 18 Status + Stress Completion

## Objective

Close the remaining Phase 18 hardening gaps by:

1. Surfacing watchdog and payload-safety health in runtime status endpoints.
2. Adding a reproducible stress benchmark profile for payload truncation overhead and event-bus pressure.

## Runtime Status Exposure

### Kernel health API

File: `agent_forge/src/agent_forge/consciousness/kernel.py`

Added runtime health methods:

- `watchdog_status()`
- `payload_safety_status()`
- `runtime_health()`

Coverage includes:

- per-module watchdog counters
- quarantine/recovery state
- watchdog thresholds
- payload safety limits and truncation-event toggle

### Trial runner status payload

File: `agent_forge/src/agent_forge/consciousness/trials.py`

`ConsciousnessTrialRunner.status()` now includes:

- `watchdog`
- `payload_safety`
- `payload_truncations_recent`
- `payload_truncation_rate_recent`

### CLI status visibility

File: `agent_forge/src/agent_forge/cli/eidctl.py`

`eidctl consciousness status` now prints watchdog and payload-safety runtime lines in non-JSON mode while JSON mode returns full structured status.

### MCP status visibility

File: `eidos_mcp/src/eidos_mcp/routers/consciousness.py`

No separate status adapter was required: MCP `consciousness_kernel_status` and `eidos://consciousness/runtime-status` both consume `runner.status()` and now carry watchdog/payload-safety fields.

## Stress Benchmark Profile

### New benchmark module

File: `agent_forge/src/agent_forge/consciousness/stress.py`

Added:

- `ConsciousnessStressBenchmark`
- `StressBenchmarkResult`
- dedicated stress emitter module for high-fanout event + broadcast pressure

Profile outputs:

- tick latency (`p50`, `p95`, `max`)
- emitted event throughput (`events/s`)
- event-bus growth bytes
- truncation count + truncation rate
- module error count
- watchdog status snapshot
- pass/fail gate bundle

### CLI commands

File: `agent_forge/src/agent_forge/cli/eidctl.py`

Added:

- `eidctl consciousness stress-benchmark`
- `eidctl consciousness latest-stress-benchmark`

### MCP commands/resources

File: `eidos_mcp/src/eidos_mcp/routers/consciousness.py`

Added tools:

- `consciousness_kernel_stress_benchmark`
- `consciousness_kernel_latest_stress_benchmark`

Added resource:

- `eidos://consciousness/runtime-latest-stress-benchmark`

## CI + Trend Aggregation

### Workflow

File: `.github/workflows/consciousness-parity.yml`

CI now executes stress benchmark in parity runs and uploads:

- `reports/consciousness_stress_benchmarks/*.json`

### Trend report

Files:

- `scripts/consciousness_benchmark_trend.py`
- `scripts/tests/test_consciousness_benchmark_trend.py`

Trend output now includes:

- stress benchmark count
- stress mean events/s
- stress mean p95 latency
- stress truncation-rate mean
- stress gate pass rate
- latest stress benchmark ID

## Tests

Added:

- `agent_forge/tests/test_consciousness_stress_benchmark.py`

Updated:

- `agent_forge/tests/test_consciousness_trials.py`
- `eidos_mcp/tests/test_mcp_tool_calls_individual.py`
- `scripts/tests/test_consciousness_benchmark_trend.py`

## Validation Commands

```sh
PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src ./eidosian_venv/bin/python -m pytest -q \
  agent_forge/tests/test_consciousness_trials.py \
  agent_forge/tests/test_consciousness_stress_benchmark.py \
  agent_forge/tests/test_consciousness_integrated_benchmark.py \
  eidos_mcp/tests/test_mcp_tool_calls_individual.py \
  scripts/tests/test_consciousness_benchmark_trend.py
```

## External References

- Circuit-breaker reliability pattern foundation:
- https://martinfowler.com/bliki/CircuitBreaker.html
- SRE reliability and observability guidance:
- https://sre.google/workbook/monitoring/
- https://sre.google/workbook/error-budget-policy/
- Security control baseline context for input validation and control assessment:
- https://csrc.nist.gov/pubs/sp/800/53/r5/final
- https://csrc.nist.gov/pubs/sp/800/53/a/r5/final
- Current NIST release update context (SP 800-53/53A Release 5.2.0, August 27, 2025):
- https://csrc.nist.gov/News/2025/nist-releases-revision-to-sp-800-53-controls
