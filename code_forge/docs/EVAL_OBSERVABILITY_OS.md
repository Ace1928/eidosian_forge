# Code Forge Eval + Observability OS

## Scope

This subsystem operationalizes evaluation as a reproducible run matrix instead of ad-hoc single runs.

Implemented in:

- `code_forge/src/code_forge/eval_os/contracts.py`
- `code_forge/src/code_forge/eval_os/tracing.py`
- `code_forge/src/code_forge/eval_os/replay.py`
- `code_forge/src/code_forge/eval_os/staleness.py`
- `code_forge/src/code_forge/eval_os/scoring.py`
- `code_forge/src/code_forge/eval_os/runner.py`
- `code_forge/src/code_forge/eval_os/taskbank.py`

CLI integration:

- `code-forge eval-init`
- `code-forge eval-run`
- `code-forge eval-staleness`

## Contract Model

### 1) TaskBank (`code_forge_taskbank_v1`)

Each task defines:

- `task_id`
- `task_type`: `swe | docs | hybrid`
- `command`
- `workdir`
- `timeout_sec`
- `contract`:
  - `require_zero_exit`
  - `required_paths`
  - `forbidden_paths`
  - `stdout_must_contain`
  - `stderr_must_not_contain`

### 2) Config Matrix (`code_forge_eval_config_matrix_v1`)

Each config defines:

- `config_id`
- `name`
- `toggles` (cache/memory/retrieval/validator/parallel flags)

Each run injects toggles into env vars as `EVAL_<TOGGLE_NAME>`.

## Run Artifact Model

Per run (`reports/code_forge_eval/runs/<run_id>/`):

- `trace.jsonl` (append-only span events with `trace_id/span_id`)
- `stdout.txt`
- `stderr.txt`
- `result.json`

Top-level:

- `reports/code_forge_eval/summary.json`
- `reports/code_forge_eval/replay_store/**`

`summary.json` includes:

- repo snapshot (`git_head`, dirty state, lockfile hashes)
- run stats (`success_rate`, p50/p95 duration, replay hits/misses)
- OTLP export stats (`attempted`, `ok`, `failed`) when `--otlp-endpoint` is configured
- per-config score vector (`success`, latency distribution, regression/staleness penalties)

## OTLP Export Wiring

`eval-run` can push per-run trace spans from `trace.jsonl` to an OTLP HTTP endpoint:

- `--otlp-endpoint http://127.0.0.1:4318`
- `--otlp-service-name code_forge_eval`
- `--otlp-timeout-sec 10`
- `--otlp-header KEY=VALUE` (repeatable)

Behavior:

- Export is best-effort and non-blocking for run success (contract pass/fail is unchanged).
- Export result is persisted under each run result (`result.json -> otlp_export`).
- Summary aggregates export outcomes under `run_stats.otlp`.

## Replay Semantics

Replay key is deterministic over:

- `task_id`
- `config_id`
- `command`
- `workdir`
- `timeout_sec`
- `env_toggles`

Modes:

- `off`: live execution only
- `record`: live execution + store deterministic replay payload
- `replay`: replay payload required; replay miss is explicit failure

When replaying into a new output directory, provide `--replay-store` so the run points at a
previous record store instead of `<output-dir>/replay_store`.

## Staleness Metrics

Input can be JSON list or JSONL of records with:

- `memory_key`
- `source_last_modified`
- `derived_at`
- `served_at`
- optional `revalidated_at`
- optional `stale_error`

Computed metrics:

- `freshness_lag_seconds` (mean/p50/p95)
- `stale_serve_rate`
- `revalidation_latency_seconds` (mean/p50/p95)
- `staleness_caused_error_rate`

## Usage

```bash
# bootstrap sample contracts
code-forge eval-init --taskbank config/eval/taskbank.json --matrix config/eval/config_matrix.json

# run eval matrix with traces and replay capture
code-forge eval-run \
  --taskbank config/eval/taskbank.json \
  --matrix config/eval/config_matrix.json \
  --output-dir reports/code_forge_eval \
  --repeats 2 \
  --max-parallel 2 \
  --replay-mode record

# replay against the captured store from a prior run
code-forge eval-run \
  --taskbank config/eval/taskbank.json \
  --matrix config/eval/config_matrix.json \
  --output-dir reports/code_forge_eval_replay \
  --replay-mode replay \
  --replay-store reports/code_forge_eval/replay_store

# compute staleness metrics from freshness records
code-forge eval-staleness \
  --input reports/code_forge_eval/freshness.jsonl \
  --output reports/code_forge_eval/staleness_metrics.json
```

## Design References

- SWE-bench benchmark and task framing:
  - https://github.com/SWE-bench/SWE-bench
  - https://www.swebench.com/SWE-bench/
- OpenTelemetry trace model:
  - https://opentelemetry.io/docs/concepts/signals/traces/
- W3C provenance model (PROV-DM):
  - https://www.w3.org/TR/prov-dm/
- HTTP stale-while-revalidate semantics:
  - https://www.rfc-editor.org/rfc/rfc5861
