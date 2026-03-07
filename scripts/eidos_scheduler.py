#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORGE_ROOT = Path(__file__).resolve().parent.parent
for extra in (FORGE_ROOT / "lib", FORGE_ROOT):
    value = str(extra)
    if extra.exists() and value not in sys.path:
        sys.path.insert(0, value)

from eidosian_core import eidosian
from eidosian_runtime import ForgeRuntimeCoordinator

RUNTIME_DIR = FORGE_ROOT / "data" / "runtime"
STATUS_PATH = RUNTIME_DIR / "eidos_scheduler_status.json"
STATE_PATH = RUNTIME_DIR / "eidos_scheduler_state.json"
PIPELINE_STATUS_PATH = RUNTIME_DIR / "living_pipeline_status.json"
_STOP_REQUESTED = False


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _pid_alive(pid: Any) -> bool:
    try:
        pid_value = int(pid or 0)
    except Exception:
        return False
    if pid_value <= 0:
        return False
    try:
        os.kill(pid_value, 0)
    except OSError:
        return False
    return True


def _load_state() -> dict[str, Any]:
    payload = _load_json(STATE_PATH)
    payload.setdefault("contract", "eidos.scheduler.state.v1")
    payload.setdefault("cycle", 0)
    payload.setdefault("consecutive_failures", 0)
    payload.setdefault("last_result", {})
    return payload


def _write_state(payload: dict[str, Any]) -> None:
    merged = {
        "contract": "eidos.scheduler.state.v1",
        "updated_at": _now_utc(),
        **payload,
    }
    _write_json(STATE_PATH, merged)


def _owner() -> str:
    return "eidos_scheduler"


def _requested_models(model: str) -> list[dict[str, Any]]:
    return [{"family": "ollama", "model": model, "role": "living_pipeline"}]


def _pipeline_pythonpath() -> str:
    paths = [
        FORGE_ROOT / "lib",
        FORGE_ROOT / "agent_forge" / "src",
        FORGE_ROOT / "memory_forge" / "src",
        FORGE_ROOT / "knowledge_forge" / "src",
        FORGE_ROOT / "code_forge" / "src",
        FORGE_ROOT / "gis_forge" / "src",
        FORGE_ROOT / "eidos_mcp" / "src",
        FORGE_ROOT / "ollama_forge" / "src",
        FORGE_ROOT / "web_interface_forge" / "src",
    ]
    existing = str(os.environ.get("PYTHONPATH") or "").strip()
    items = [str(path) for path in paths if path.exists()]
    if existing:
        items.append(existing)
    return ":".join(items)


def _status_payload(
    *,
    state: str,
    current_task: str,
    interval_sec: float,
    cycle: int,
    next_run_in_seconds: float = 0.0,
    last_result: dict[str, Any] | None = None,
    last_error: str = "",
    pid: int | None = None,
) -> dict[str, Any]:
    return {
        "contract": "eidos.scheduler.status.v1",
        "updated_at": _now_utc(),
        "state": state,
        "owner": _owner(),
        "current_task": current_task,
        "cycle": int(cycle),
        "interval_sec": float(interval_sec),
        "next_run_in_seconds": max(0.0, float(next_run_in_seconds)),
        "pid": int(pid or os.getpid()),
        "last_result": dict(last_result or {}),
        "last_error": str(last_error or ""),
    }


def _mark_scheduler_state(
    *,
    cycle: int,
    interval_sec: float,
    model: str,
    run_graphrag: bool,
    code_max_files: int | None,
    state: str,
    last_result: dict[str, Any] | None = None,
    stop_requested: bool = False,
) -> None:
    previous = _load_state()
    consecutive_failures = int(previous.get("consecutive_failures") or 0)
    if state == "success":
        consecutive_failures = 0
    elif state in {"error", "timeout"}:
        consecutive_failures += 1
    _write_state(
        {
            "cycle": int(cycle),
            "state": state,
            "interval_sec": float(interval_sec),
            "model": model,
            "run_graphrag": bool(run_graphrag),
            "code_max_files": code_max_files,
            "consecutive_failures": consecutive_failures,
            "last_result": dict(last_result or previous.get("last_result") or {}),
            "stop_requested": bool(stop_requested),
        }
    )


@eidosian()
def run_scheduler_cycle(
    *,
    interval_sec: float,
    timeout_sec: float,
    run_graphrag: bool,
    code_max_files: int | None,
    repo_root: Path,
    output_root: Path,
    workspace_root: Path,
    model: str,
    coordinator: ForgeRuntimeCoordinator | None = None,
    python_bin: str | None = None,
    cycle: int = 1,
) -> dict[str, Any]:
    coordinator = coordinator or ForgeRuntimeCoordinator()
    if _STOP_REQUESTED:
        result = {"status": "stopped", "reason": "stop_requested"}
        _write_json(
            STATUS_PATH,
            _status_payload(
                state="stopped",
                current_task="living_pipeline",
                interval_sec=interval_sec,
                cycle=cycle,
                next_run_in_seconds=0.0,
                last_result=result,
            ),
        )
        _mark_scheduler_state(
            cycle=cycle,
            interval_sec=interval_sec,
            model=model,
            run_graphrag=run_graphrag,
            code_max_files=code_max_files,
            state="stopped",
            last_result=result,
            stop_requested=True,
        )
        return result
    decision = coordinator.can_allocate(
        owner=_owner(), requested_models=_requested_models(model), allow_same_owner=False
    )
    if not decision.get("allowed"):
        payload = _status_payload(
            state="waiting",
            current_task="living_pipeline",
            interval_sec=interval_sec,
            cycle=cycle,
            next_run_in_seconds=interval_sec,
            last_result={"status": "waiting", "reason": str(decision.get("reason") or "blocked")},
        )
        _write_json(STATUS_PATH, payload)
        _mark_scheduler_state(
            cycle=cycle,
            interval_sec=interval_sec,
            model=model,
            run_graphrag=run_graphrag,
            code_max_files=code_max_files,
            state="waiting",
            last_result=payload.get("last_result"),
        )
        return payload

    coordinator.heartbeat(
        owner=_owner(),
        task="living_pipeline",
        state="running",
        active_models=_requested_models(model),
        metadata={
            "mode": "scheduler_cycle",
            "cycle": int(cycle),
            "interval_sec": float(interval_sec),
            "doc_model": model,
        },
    )
    start = time.perf_counter()
    _write_json(
        STATUS_PATH,
        _status_payload(
            state="running",
            current_task="living_pipeline",
            interval_sec=interval_sec,
            cycle=cycle,
            next_run_in_seconds=0.0,
        ),
    )
    _mark_scheduler_state(
        cycle=cycle,
        interval_sec=interval_sec,
        model=model,
        run_graphrag=run_graphrag,
        code_max_files=code_max_files,
        state="running",
    )
    command = [
        python_bin or str(FORGE_ROOT / "eidosian_venv" / "bin" / "python"),
        str(FORGE_ROOT / "scripts" / "living_knowledge_pipeline.py"),
        "--repo-root",
        str(repo_root),
        "--output-root",
        str(output_root),
        "--workspace-root",
        str(workspace_root),
    ]
    if run_graphrag:
        command.append("--run-graphrag")
    if code_max_files is not None:
        command.extend(["--code-max-files", str(code_max_files)])

    try:
        proc = subprocess.run(
            command,
            cwd=str(FORGE_ROOT),
            capture_output=True,
            text=True,
            timeout=max(60.0, float(timeout_sec)),
            env={
                **os.environ,
                "EIDOS_QWEN_MODEL": model,
                "EIDOS_FORGE_ROOT": str(FORGE_ROOT),
                "PYTHONPATH": _pipeline_pythonpath(),
            },
        )
        elapsed = round(time.perf_counter() - start, 3)
        stdout = str(proc.stdout or "").strip()
        parsed: dict[str, Any] = {}
        if stdout:
            try:
                parsed = json.loads(stdout)
            except Exception:
                parsed = {"stdout": stdout[:1200]}
        result = {
            "status": "success" if proc.returncode == 0 else "error",
            "returncode": int(proc.returncode),
            "elapsed_sec": elapsed,
            "run_id": str(parsed.get("run_id") or ""),
            "records_total": int(parsed.get("records_total") or 0),
            "stdout": stdout[:1200],
            "stderr": str(proc.stderr or "").strip()[:1200],
            "pipeline_status": _load_json(PIPELINE_STATUS_PATH),
        }
        _write_json(
            STATUS_PATH,
            _status_payload(
                state="sleeping" if proc.returncode == 0 else "error",
                current_task="living_pipeline",
                interval_sec=interval_sec,
                cycle=cycle,
                next_run_in_seconds=interval_sec,
                last_result=result,
                last_error=result["stderr"] if proc.returncode != 0 else "",
            ),
        )
        _mark_scheduler_state(
            cycle=cycle,
            interval_sec=interval_sec,
            model=model,
            run_graphrag=run_graphrag,
            code_max_files=code_max_files,
            state=result["status"],
            last_result=result,
        )
        return result
    except subprocess.TimeoutExpired as exc:
        elapsed = round(time.perf_counter() - start, 3)
        result = {
            "status": "timeout",
            "returncode": -1,
            "elapsed_sec": elapsed,
            "stdout": str(exc.stdout or "")[:1200],
            "stderr": str(exc.stderr or "")[:1200],
        }
        _write_json(
            STATUS_PATH,
            _status_payload(
                state="timeout",
                current_task="living_pipeline",
                interval_sec=interval_sec,
                cycle=cycle,
                next_run_in_seconds=interval_sec,
                last_result=result,
                last_error="pipeline_timeout",
            ),
        )
        _mark_scheduler_state(
            cycle=cycle,
            interval_sec=interval_sec,
            model=model,
            run_graphrag=run_graphrag,
            code_max_files=code_max_files,
            state="timeout",
            last_result=result,
        )
        return result
    finally:
        coordinator.clear_owner(
            _owner(),
            metadata={
                "exclusive": False,
                "mode": "scheduler_cycle",
                "task": "living_pipeline",
                "released_reason": "cycle_finished",
            },
        )


@eidosian()
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coordinator-aware living pipeline scheduler.")
    parser.add_argument(
        "--interval-sec", type=float, default=float(os.environ.get("EIDOS_SCHEDULER_INTERVAL_SEC", 900))
    )
    parser.add_argument("--timeout-sec", type=float, default=float(os.environ.get("EIDOS_SCHEDULER_TIMEOUT_SEC", 7200)))
    parser.add_argument("--max-cycles", type=int, default=int(os.environ.get("EIDOS_SCHEDULER_MAX_CYCLES", 0)))
    parser.add_argument("--code-max-files", type=int, default=None)
    parser.add_argument("--repo-root", default=str(FORGE_ROOT))
    parser.add_argument("--output-root", default=str(FORGE_ROOT / "reports" / "living_knowledge"))
    parser.add_argument("--workspace-root", default=str(FORGE_ROOT / "data" / "living_knowledge" / "workspace"))
    parser.add_argument("--model", default=os.environ.get("EIDOS_QWEN_MODEL", "qwen3.5:2b"))
    parser.add_argument("--run-graphrag", action="store_true")
    return parser.parse_args()


@eidosian()
def main() -> int:
    global _STOP_REQUESTED
    args = parse_args()
    interval_sec = max(5.0, float(args.interval_sec))
    timeout_sec = max(60.0, float(args.timeout_sec))
    max_cycles = max(0, int(args.max_cycles))
    prior_status = _load_json(STATUS_PATH)
    if str(prior_status.get("state") or "") == "running" and not _pid_alive(prior_status.get("pid")):
        _write_json(
            STATUS_PATH,
            _status_payload(
                state="recovered",
                current_task=str(prior_status.get("current_task") or "living_pipeline"),
                interval_sec=interval_sec,
                cycle=int(prior_status.get("cycle") or 0),
                next_run_in_seconds=0.0,
                last_result=prior_status.get("last_result") if isinstance(prior_status.get("last_result"), dict) else {},
                last_error="stale_running_status_recovered",
            ),
        )
    cycle = int(_load_state().get("cycle") or 0)

    def _request_stop(_signum, _frame) -> None:
        global _STOP_REQUESTED
        _STOP_REQUESTED = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _request_stop)

    while True:
        if _STOP_REQUESTED:
            _write_json(
                STATUS_PATH,
                _status_payload(
                    state="stopped",
                    current_task="living_pipeline",
                    interval_sec=interval_sec,
                    cycle=cycle,
                    next_run_in_seconds=0.0,
                    last_result=_load_state().get("last_result") if isinstance(_load_state().get("last_result"), dict) else {},
                    last_error="stop_requested",
                ),
            )
            _mark_scheduler_state(
                cycle=cycle,
                interval_sec=interval_sec,
                model=str(args.model),
                run_graphrag=bool(args.run_graphrag),
                code_max_files=args.code_max_files,
                state="stopped",
                last_result=_load_state().get("last_result") if isinstance(_load_state().get("last_result"), dict) else {},
                stop_requested=True,
            )
            return 0
        cycle += 1
        run_scheduler_cycle(
            interval_sec=interval_sec,
            timeout_sec=timeout_sec,
            run_graphrag=bool(args.run_graphrag),
            code_max_files=args.code_max_files,
            repo_root=Path(args.repo_root).resolve(),
            output_root=Path(args.output_root).resolve(),
            workspace_root=Path(args.workspace_root).resolve(),
            model=str(args.model),
            cycle=cycle,
        )
        if max_cycles and cycle >= max_cycles:
            return 0
        sleep_remaining = interval_sec
        while sleep_remaining > 0:
            if _STOP_REQUESTED:
                break
            current_state = _load_state()
            _write_json(
                STATUS_PATH,
                _status_payload(
                    state="sleeping",
                    current_task="living_pipeline",
                    interval_sec=interval_sec,
                    cycle=cycle,
                    next_run_in_seconds=sleep_remaining,
                    last_result=current_state.get("last_result") if isinstance(current_state.get("last_result"), dict) else {},
                    last_error=_load_json(STATUS_PATH).get("last_error") if STATUS_PATH.exists() else "",
                ),
            )
            step = min(5.0, sleep_remaining)
            time.sleep(step)
            sleep_remaining = max(0.0, sleep_remaining - step)


if __name__ == "__main__":
    raise SystemExit(main())
