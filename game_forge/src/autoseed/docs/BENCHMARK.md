# Benchmark Guide

## Quick Run

- `npm run benchmark`

This builds the project and runs a tick-loop benchmark using the compiled core simulation.

## Configuration

Override defaults via environment variables:

- `TICKS` (default 2000)
- `SYSTEMS` (default 7)
- `SEED` (default 4242)

Example:

```
TICKS=10000 SYSTEMS=12 npm run benchmark
```

## Notes

- Benchmarks measure simulation tick throughput only (no rendering).
- Run on a quiet machine for consistent numbers.
