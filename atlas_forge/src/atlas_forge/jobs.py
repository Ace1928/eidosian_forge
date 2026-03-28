import os
import subprocess
import threading
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
from asyncio import to_thread
from fastapi import HTTPException
from .config import (
    FORGE_ROOT, SERVICES_SCRIPT, SERVICE_ACTION_LOG, 
    SCHEDULER_CONTROL_SCRIPT, RUNTIME_DIR
)
from .utils import _now_utc_iso, _write_json, _read_json

logger = logging.getLogger("eidos_dashboard")

def _build_forge_subprocess_env() -> Dict[str, str]:
    from .config import setup_pythonpath
    setup_pythonpath()
    env = os.environ.copy()
    env["EIDOS_FORGE_ROOT"] = str(FORGE_ROOT)
    return env

def _run_service_action_async(action: str, service: str | None = None) -> None:
    env = _build_forge_subprocess_env()
    SERVICE_ACTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    command = [str(SERVICES_SCRIPT), action]
    if service:
        command.append(service)
    with SERVICE_ACTION_LOG.open("a", encoding="utf-8") as handle:
        subprocess.Popen(
            command,
            cwd=str(FORGE_ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )

async def _service_command(action: str, service: str | None = None) -> Dict[str, Any]:
    allowed = {"start", "stop", "pause", "resume", "restart", "status", "low-load", "restore-standard", "shell-reset"}
    if action not in allowed:
        raise HTTPException(status_code=400, detail="Invalid service action")
    if not SERVICES_SCRIPT.exists():
        raise HTTPException(status_code=503, detail="Service controller unavailable")
    
    allowed_services = {None, "", "all", "ollama-qwen", "ollama-embedding", "mcp", "doc-forge", "atlas", "scheduler", "local-agent"}
    service = (service or "").strip() or None
    if service not in allowed_services:
        raise HTTPException(status_code=400, detail="Invalid service target")
        
    env = _build_forge_subprocess_env()
    if action == "status":
        env["EIDOS_SKIP_ATLAS_HEALTH_CHECK"] = "1"
        
    if action != "status":
        await to_thread(_run_service_action_async, action, service)
        return {
            "action": action,
            "service": service or "all",
            "accepted": True,
            "queued": True,
            "ok": True,
        }
        
    result = subprocess.run(
        [str(SERVICES_SCRIPT), action, *( [service] if service else [] )],
        cwd=str(FORGE_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    return {
        "action": action,
        "service": service or "all",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "ok": result.returncode == 0,
        "services": _parse_service_status_output(result.stdout) if action == "status" else [],
    }

def _parse_service_status_output(raw: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or ":" not in line or line.startswith("Interactive shell refcount"):
            continue
        name, state = line.split(":", 1)
        state_value = state.strip()
        rows.append({
            "name": name.strip(),
            "state": state_value,
            "running": ("run:" in state_value or "running(" in state_value) and "paused" not in state_value,
            "paused": "paused" in state_value,
        })
    return rows

def _scheduler_command(action: str) -> Dict[str, Any]:
    if not SCHEDULER_CONTROL_SCRIPT.exists():
        raise HTTPException(status_code=503, detail="Scheduler controller unavailable")
    env = _build_forge_subprocess_env()
    result = subprocess.run(
        [str(FORGE_ROOT / "eidosian_venv" / "bin" / "python"), str(SCHEDULER_CONTROL_SCRIPT), action],
        cwd=str(FORGE_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    payload = {}
    try:
        payload = json.loads(result.stdout or "{}")
    except Exception:
        pass
    return {
        "action": action,
        "ok": result.returncode == 0,
        "payload": payload,
    }

# --- Background Job Runners ---
# (These would be similar to the ones in main.py, e.g., _run_docs_upsert_batch_job)
# ...
