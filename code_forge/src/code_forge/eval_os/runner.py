from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex
from typing import Any

from code_forge.eval_os.contracts import (
    EvalConfig,
    load_eval_config_matrix,
    load_taskbank,
)
from code_forge.eval_os.replay import ReplayStore, build_replay_key
from code_forge.eval_os.scoring import score_runs
from code_forge.eval_os.staleness import (
    compute_staleness_metrics,
    load_freshness_records,
)
from code_forge.eval_os.tracing import (
    TraceMeta,
    TraceRecorder,
    export_trace_jsonl_to_otlp,
    new_trace_id,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return float(ordered[max(0, min(idx, len(ordered) - 1))])


def _collect_repo_snapshot(repo_root: Path) -> dict[str, Any]:
    def _run_git(*args: str) -> str:
        try:
            out = subprocess.check_output(
                ["git", "-C", str(repo_root), *args],
                stderr=subprocess.STDOUT,
                text=True,
            )
            return out.strip()
        except Exception:
            return ""

    lock_files = [
        "requirements.txt",
        "pyproject.toml",
        "poetry.lock",
        "package-lock.json",
        "pnpm-lock.yaml",
        "Cargo.lock",
        "uv.lock",
    ]
    lock_hashes = {}
    for rel in lock_files:
        path = repo_root / rel
        if not path.exists() or not path.is_file():
            continue
        try:
            lock_hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            continue

    return {
        "repo_root": str(repo_root),
        "git_head": _run_git("rev-parse", "HEAD"),
        "git_branch": _run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_run_git("status", "--porcelain")),
        "lock_hashes": lock_hashes,
    }


def _render_toggle_env(config: EvalConfig) -> dict[str, str]:
    env = {}
    for key, value in sorted(config.toggles.items()):
        env_key = f"EVAL_{str(key).upper()}"
        if isinstance(value, bool):
            env[env_key] = "1" if value else "0"
        else:
            env[env_key] = str(value)
    env["EVAL_CONFIG_ID"] = config.config_id
    env["EVAL_CONFIG_NAME"] = config.name
    return env


@dataclass(frozen=True)
class EvalRunOptions:
    taskbank_path: Path
    config_matrix_path: Path
    output_dir: Path
    repo_root: Path
    repeats: int = 1
    replay_mode: str = "off"  # off | record | replay
    max_parallel: int = 1
    default_timeout_sec: int = 1200
    staleness_log_path: Path | None = None
    replay_store_path: Path | None = None
    otlp_endpoint: str | None = None
    otlp_service_name: str = "code_forge_eval"
    otlp_timeout_sec: int = 10
    otlp_headers: dict[str, str] | None = None


def _execute_single_run(
    *,
    task: Any,
    config: EvalConfig,
    attempt: int,
    options: EvalRunOptions,
    repo_snapshot: dict[str, Any],
    replay_store: ReplayStore,
) -> dict[str, Any]:
    run_id = f"eval_{task.task_id}_{config.config_id}_{attempt}_{token_hex(4)}"
    run_dir = options.output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trace_meta = TraceMeta(
        run_id=run_id,
        task_id=task.task_id,
        config_id=config.config_id,
        trace_id=new_trace_id(),
    )
    trace_path = run_dir / "trace.jsonl"
    recorder = TraceRecorder(trace_path, meta=trace_meta)

    workdir = (options.repo_root / task.workdir).resolve()
    timeout_sec = max(1, int(task.timeout_sec or options.default_timeout_sec))
    toggle_env = _render_toggle_env(config)
    replay_key = build_replay_key(
        task_id=task.task_id,
        config_id=config.config_id,
        command=task.command,
        workdir=str(workdir),
        timeout_sec=timeout_sec,
        env_toggles=toggle_env,
    )

    recorder.emit(
        event_type="run.start",
        span_id=None,
        parent_span_id=None,
        name="eval_run",
        attributes={
            "attempt": attempt,
            "replay_mode": options.replay_mode,
            "workdir": str(workdir),
            "task_type": task.task_type,
            "timeout_sec": timeout_sec,
            "replay_key": replay_key,
        },
        status="start",
    )

    used_replay = False
    started = time.perf_counter()
    stdout_text = ""
    stderr_text = ""
    returncode = 1
    timed_out = False
    replay_miss = False

    with recorder.span(
        name="command.execute",
        event_type="command",
        attributes={"command": task.command},
    ):
        if options.replay_mode == "replay":
            replay_payload = replay_store.load(replay_key)
            if replay_payload is None:
                replay_miss = True
                returncode = 125
                stderr_text = f"replay miss for key={replay_key}"
            else:
                used_replay = True
                returncode = int(replay_payload.get("returncode", 1))
                stdout_text = str(replay_payload.get("stdout", ""))
                stderr_text = str(replay_payload.get("stderr", ""))
                timed_out = bool(replay_payload.get("timed_out", False))
        else:
            env = os.environ.copy()
            env.update(toggle_env)
            try:
                proc = subprocess.run(
                    ["/bin/sh", "-lc", task.command],
                    cwd=str(workdir),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    check=False,
                )
                returncode = int(proc.returncode)
                stdout_text = proc.stdout or ""
                stderr_text = proc.stderr or ""
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                returncode = 124
                stdout_text = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
                stderr_text = (exc.stderr or "") if isinstance(exc.stderr, str) else ""

    duration_ms = (time.perf_counter() - started) * 1000.0

    (run_dir / "stdout.txt").write_text(stdout_text, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(stderr_text, encoding="utf-8")

    contract_result = task.contract.evaluate(
        workdir=workdir,
        returncode=returncode,
        stdout=stdout_text,
        stderr=stderr_text,
    )
    success = bool(contract_result.get("pass"))

    payload = {
        "run_id": run_id,
        "task_id": task.task_id,
        "task_type": task.task_type,
        "task_hash": task.task_hash,
        "config_id": config.config_id,
        "config_name": config.name,
        "attempt": attempt,
        "started_at": _utc_now(),
        "duration_ms": duration_ms,
        "returncode": returncode,
        "timed_out": timed_out,
        "success": success,
        "replay_used": used_replay,
        "replay_miss": replay_miss,
        "replay_key": replay_key,
        "trace_path": str(trace_path),
        "stdout_path": str(run_dir / "stdout.txt"),
        "stderr_path": str(run_dir / "stderr.txt"),
        "stdout_sha256": _sha256_text(stdout_text),
        "stderr_sha256": _sha256_text(stderr_text),
        "stdout_bytes": len(stdout_text.encode("utf-8")),
        "stderr_bytes": len(stderr_text.encode("utf-8")),
        "contract": contract_result,
        "repo_snapshot": repo_snapshot,
        "toggle_env": toggle_env,
    }

    if options.otlp_endpoint:
        otlp_result = export_trace_jsonl_to_otlp(
            trace_path=trace_path,
            endpoint=options.otlp_endpoint,
            service_name=options.otlp_service_name,
            timeout_sec=options.otlp_timeout_sec,
            headers=options.otlp_headers,
            resource_attributes={
                "code_forge.run_id": run_id,
                "code_forge.task_id": task.task_id,
                "code_forge.config_id": config.config_id,
            },
        )
        payload["otlp_export"] = otlp_result

    if options.replay_mode == "record":
        replay_store.save(
            replay_key,
            {
                "returncode": returncode,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "timed_out": timed_out,
                "recorded_at": _utc_now(),
            },
        )

    (run_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    recorder.emit(
        event_type="run.end",
        span_id=None,
        parent_span_id=None,
        name="eval_run",
        attributes={
            "success": success,
            "duration_ms": duration_ms,
            "returncode": returncode,
            "replay_used": used_replay,
            "replay_miss": replay_miss,
        },
        status="ok" if success else "error",
    )
    return payload


def run_eval_suite(options: EvalRunOptions) -> dict[str, Any]:
    options = EvalRunOptions(
        taskbank_path=Path(options.taskbank_path).resolve(),
        config_matrix_path=Path(options.config_matrix_path).resolve(),
        output_dir=Path(options.output_dir).resolve(),
        repo_root=Path(options.repo_root).resolve(),
        repeats=max(1, int(options.repeats)),
        replay_mode=str(options.replay_mode).strip().lower(),
        max_parallel=max(1, int(options.max_parallel)),
        default_timeout_sec=max(1, int(options.default_timeout_sec)),
        staleness_log_path=(Path(options.staleness_log_path).resolve() if options.staleness_log_path else None),
        replay_store_path=(Path(options.replay_store_path).resolve() if options.replay_store_path else None),
        otlp_endpoint=(str(options.otlp_endpoint).strip() if options.otlp_endpoint else None),
        otlp_service_name=str(options.otlp_service_name or "code_forge_eval"),
        otlp_timeout_sec=max(1, int(options.otlp_timeout_sec)),
        otlp_headers=dict(options.otlp_headers or {}),
    )
    if options.replay_mode not in {"off", "record", "replay"}:
        raise ValueError("replay_mode must be one of: off, record, replay")

    options.output_dir.mkdir(parents=True, exist_ok=True)

    taskbank_schema, tasks, taskbank_meta = load_taskbank(options.taskbank_path)
    matrix = load_eval_config_matrix(options.config_matrix_path)
    repo_snapshot = _collect_repo_snapshot(options.repo_root)
    replay_store_root = (
        options.replay_store_path if options.replay_store_path is not None else (options.output_dir / "replay_store")
    )
    replay_store = ReplayStore(replay_store_root)

    run_specs: list[tuple[Any, EvalConfig, int]] = []
    for task in tasks:
        for config in matrix.configs:
            for attempt in range(1, options.repeats + 1):
                run_specs.append((task, config, attempt))

    run_results: list[dict[str, Any]] = []
    if options.max_parallel == 1:
        for task, config, attempt in run_specs:
            run_results.append(
                _execute_single_run(
                    task=task,
                    config=config,
                    attempt=attempt,
                    options=options,
                    repo_snapshot=repo_snapshot,
                    replay_store=replay_store,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=options.max_parallel) as pool:
            futures = {
                pool.submit(
                    _execute_single_run,
                    task=task,
                    config=config,
                    attempt=attempt,
                    options=options,
                    repo_snapshot=repo_snapshot,
                    replay_store=replay_store,
                ): (task.task_id, config.config_id, attempt)
                for (task, config, attempt) in run_specs
            }
            for fut in as_completed(futures):
                run_results.append(fut.result())

    run_results.sort(
        key=lambda row: (
            row.get("task_id"),
            row.get("config_id"),
            row.get("attempt", 0),
        )
    )

    config_scores: dict[str, dict[str, Any]] = {}
    for config in matrix.configs:
        selected = [row for row in run_results if row.get("config_id") == config.config_id]
        config_scores[config.config_id] = {
            "config_name": config.name,
            "score": score_runs(selected),
        }

    durations = [float(row.get("duration_ms") or 0.0) for row in run_results]
    success_count = sum(1 for row in run_results if bool(row.get("success")))
    replay_hits = sum(1 for row in run_results if bool(row.get("replay_used")))
    replay_misses = sum(1 for row in run_results if bool(row.get("replay_miss")))
    otlp_attempted = sum(1 for row in run_results if isinstance(row.get("otlp_export"), dict))
    otlp_ok = sum(1 for row in run_results if bool((row.get("otlp_export") or {}).get("ok")))

    staleness_payload = None
    if options.staleness_log_path is not None and options.staleness_log_path.exists():
        rows = load_freshness_records(options.staleness_log_path)
        staleness_payload = compute_staleness_metrics(rows)

    summary = {
        "schema_version": "code_forge_eval_report_v1",
        "generated_at": _utc_now(),
        "taskbank": {
            "path": str(options.taskbank_path),
            "schema_version": taskbank_schema,
            "metadata": taskbank_meta,
            "task_count": len(tasks),
        },
        "config_matrix": {
            "path": str(options.config_matrix_path),
            "schema_version": matrix.schema_version,
            "metadata": matrix.metadata,
            "config_count": len(matrix.configs),
        },
        "options": {
            "repeats": options.repeats,
            "replay_mode": options.replay_mode,
            "max_parallel": options.max_parallel,
            "default_timeout_sec": options.default_timeout_sec,
            "replay_store_path": str(replay_store_root),
            "otlp_endpoint": options.otlp_endpoint,
            "otlp_service_name": options.otlp_service_name,
            "otlp_timeout_sec": options.otlp_timeout_sec,
        },
        "repo_snapshot": repo_snapshot,
        "run_stats": {
            "run_count": len(run_results),
            "success_count": success_count,
            "success_rate": success_count / max(1, len(run_results)),
            "duration_ms": {
                "mean": (sum(durations) / len(durations)) if durations else 0.0,
                "p50": _percentile(durations, 50),
                "p95": _percentile(durations, 95),
            },
            "replay_hits": replay_hits,
            "replay_misses": replay_misses,
            "otlp": {
                "enabled": bool(options.otlp_endpoint),
                "attempted": otlp_attempted,
                "ok": otlp_ok,
                "failed": max(0, otlp_attempted - otlp_ok),
            },
        },
        "config_scores": config_scores,
        "staleness": staleness_payload,
        "runs": run_results,
    }

    summary_path = options.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
