#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx

FORGE_ROOT = Path(__file__).resolve().parent.parent
for extra in (FORGE_ROOT / "lib", FORGE_ROOT / "eidos_mcp" / "src"):
    value = str(extra)
    if extra.exists() and value not in sys.path:
        sys.path.insert(0, value)

from eidosian_core.ports import get_service_url
from eidosian_runtime import ForgeRuntimeCoordinator

RUNTIME_DIR = FORGE_ROOT / "data" / "runtime"
SCHEDULER_STATUS_PATH = RUNTIME_DIR / "eidos_scheduler_status.json"
PIPELINE_STATUS_PATH = RUNTIME_DIR / "living_pipeline_status.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _eta_hint() -> tuple[str, float | None]:
    pipeline = _load_json(PIPELINE_STATUS_PATH)
    scheduler = _load_json(SCHEDULER_STATUS_PATH)
    phase = str(pipeline.get("phase") or scheduler.get("current_task") or "").strip()
    owner = str((scheduler.get("owner") or "")).strip()
    eta = pipeline.get("eta_seconds")
    if eta is None:
        eta = scheduler.get("next_run_in_seconds")
    try:
        eta_value = None if eta is None else max(0.0, float(eta))
    except Exception:
        eta_value = None
    label = phase or owner or "current task"
    return label, eta_value


def _wait_message(decision: dict[str, Any], coordinator_payload: dict[str, Any]) -> str:
    owner = str(
        decision.get("exclusive_owner") or decision.get("active_owner") or coordinator_payload.get("owner") or ""
    ).strip()
    task = str(coordinator_payload.get("task") or "current task").strip()
    label, eta = _eta_hint()
    task_label = label or task or "current task"
    if eta is None:
        return f"waiting for {owner or 'active worker'} to release {task_label}"
    eta_text = f"{int(round(eta))}s"
    return f"waiting for {owner or 'active worker'} to release {task_label} (eta ~{eta_text})"


def _lease_metadata(model: str) -> dict[str, Any]:
    return {
        "exclusive": True,
        "exclusive_owner": "qwenchat",
        "mode": "interactive_chat",
        "model": model,
        "chat_url": get_service_url(
            "ollama_qwen_http", default_port=8938, default_host="127.0.0.1", default_path=""
        ).rstrip("/"),
    }


def acquire_chat_lease(
    coordinator: ForgeRuntimeCoordinator,
    *,
    model: str,
    poll_interval: float,
    out: Any = sys.stderr,
) -> str:
    owner = "qwenchat"
    requested_models = [{"family": "ollama", "model": model, "role": "interactive_chat"}]
    last_message = ""
    while True:
        decision = coordinator.can_allocate(owner=owner, requested_models=requested_models, allow_same_owner=False)
        if decision.get("allowed"):
            coordinator.heartbeat(
                owner=owner,
                task="interactive_chat",
                state="interactive",
                active_models=requested_models,
                metadata=_lease_metadata(model),
            )
            return owner
        payload = coordinator.read()
        message = _wait_message(decision, payload)
        if message != last_message:
            print(f"[qwenchat] {message}", file=out)
            last_message = message
        time.sleep(max(1.0, float(poll_interval)))


def _heartbeat_loop(
    coordinator: ForgeRuntimeCoordinator, *, owner: str, model: str, stop_event: threading.Event
) -> None:
    requested_models = [{"family": "ollama", "model": model, "role": "interactive_chat"}]
    while not stop_event.wait(5.0):
        coordinator.heartbeat(
            owner=owner,
            task="interactive_chat",
            state="interactive",
            active_models=requested_models,
            metadata=_lease_metadata(model),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive qwen chat against the dedicated Ollama service.")
    parser.add_argument("--model", default=os.environ.get("EIDOS_QWEN_MODEL", "qwen3.5:2b"))
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--prompt", default="", help="Run a single prompt and exit instead of interactive chat.")
    parser.add_argument("--timeout-sec", type=float, default=120.0, help="Timeout for one-shot prompt mode.")
    parser.add_argument("extra", nargs="*", help="Additional arguments passed to `ollama run`.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    coordinator = ForgeRuntimeCoordinator()
    owner = acquire_chat_lease(coordinator, model=str(args.model), poll_interval=float(args.poll_interval))
    stop_event = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(coordinator,),
        kwargs={"owner": owner, "model": str(args.model), "stop_event": stop_event},
        daemon=True,
    )
    heartbeat.start()

    base_url = get_service_url("ollama_qwen_http", default_port=8938, default_host="127.0.0.1", default_path="").rstrip(
        "/"
    )
    env = os.environ.copy()
    env["OLLAMA_HOST"] = base_url
    print(f"[qwenchat] using {base_url} model={args.model}", file=sys.stderr)

    proc: subprocess.Popen[str] | None = None

    def _forward(signum: int, _frame: Any) -> None:
        if proc is not None and proc.poll() is None:
            proc.send_signal(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _forward)

    try:
        if args.prompt.strip():
            payload = {
                "model": str(args.model),
                "prompt": str(args.prompt),
                "stream": False,
                "options": {"num_predict": 256, "temperature": 0.2},
                "think": False,
            }
            try:
                with httpx.Client(timeout=max(5.0, float(args.timeout_sec))) as client:
                    response = client.post(f"{base_url}/api/generate", json=payload)
                    response.raise_for_status()
                    body = response.json()
                print(str(body.get("response") or "").strip())
                return_code = 0
            except httpx.TimeoutException:
                print(f"[qwenchat] request timed out after {float(args.timeout_sec):.0f}s", file=sys.stderr)
                return_code = 1
            except httpx.HTTPError as exc:
                print(f"[qwenchat] request failed: {exc}", file=sys.stderr)
                return_code = 1
        else:
            cmd = ["ollama", "run", str(args.model), *list(args.extra)]
            proc = subprocess.Popen(cmd, env=env)
            return_code = proc.wait()
    finally:
        stop_event.set()
        heartbeat.join(timeout=1.0)
        coordinator.clear_owner(owner, metadata={"exclusive": False, "exclusive_owner": "", "task": "interactive_chat"})

    return int(return_code)


if __name__ == "__main__":
    raise SystemExit(main())
