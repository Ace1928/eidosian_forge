import atexit
import fcntl
import json
import logging
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import termios
import threading
import uuid
from asyncio import to_thread
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import markdown
import psutil
from eidosian_runtime import collect_runtime_capabilities
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --- Configuration ---
FORGE_ROOT = Path(os.environ.get("EIDOS_FORGE_ROOT", "/data/data/com.termux/files/home/eidosian_forge")).resolve()
FORGE_IMPORT_PATHS = (
    FORGE_ROOT / "web_interface_forge" / "src",
    FORGE_ROOT / "lib",
    FORGE_ROOT / "doc_forge" / "src",
    FORGE_ROOT / "agent_forge" / "src",
    FORGE_ROOT / "code_forge" / "src",
    FORGE_ROOT / "file_forge" / "src",
    FORGE_ROOT / "gis_forge" / "src",
    FORGE_ROOT / "memory_forge" / "src",
    FORGE_ROOT / "narrative_forge" / "src",
    FORGE_ROOT / "eidos_mcp" / "src",
    FORGE_ROOT / "neural_forge" / "src",
)
for extra in FORGE_IMPORT_PATHS:
    extra_text = str(extra)
    if extra.exists() and extra_text not in sys.path:
        sys.path.insert(0, extra_text)

from file_forge import FileForge  # type: ignore
from file_forge.library import FileLibraryDB  # type: ignore

DOC_RUNTIME = FORGE_ROOT / "doc_forge" / "runtime"
DOC_FINAL = DOC_RUNTIME / "final_docs"
DOC_INDEX = DOC_RUNTIME / "doc_index.json"
DOC_STATUS = DOC_RUNTIME / "processor_status.json"
DOC_HISTORY = DOC_RUNTIME / "processor_history.jsonl"
RUNTIME_DIR = FORGE_ROOT / "data" / "runtime"
HOME_ROOT = Path(os.environ.get("HOME", "/data/data/com.termux/files/home")).resolve()
LOCAL_AGENT_STATUS = RUNTIME_DIR / "local_mcp_agent" / "status.json"
LOCAL_AGENT_HISTORY = RUNTIME_DIR / "local_mcp_agent" / "history.jsonl"
QWENCHAT_STATUS = RUNTIME_DIR / "qwenchat" / "status.json"
QWENCHAT_HISTORY = RUNTIME_DIR / "qwenchat" / "history.jsonl"
SCHEDULER_STATUS = RUNTIME_DIR / "eidos_scheduler_status.json"
SCHEDULER_HISTORY = RUNTIME_DIR / "eidos_scheduler_history.jsonl"
LIVING_PIPELINE_STATUS = RUNTIME_DIR / "living_pipeline_status.json"
LIVING_PIPELINE_HISTORY = RUNTIME_DIR / "living_pipeline_history.jsonl"
COORDINATOR_STATUS = RUNTIME_DIR / "forge_coordinator_status.json"
COORDINATOR_HISTORY = RUNTIME_DIR / "forge_runtime_trends.json"
BOOT_STATUS = RUNTIME_DIR / "termux_boot_status.json"
CAPABILITIES_STATUS = RUNTIME_DIR / "platform_capabilities.json"
DIRECTORY_DOCS_STATUS = RUNTIME_DIR / "directory_docs_status.json"
DIRECTORY_DOCS_HISTORY = RUNTIME_DIR / "directory_docs_history.json"
DIRECTORY_DOCS_TREE = RUNTIME_DIR / "directory_docs_tree.json"
DOCS_BATCH_STATUS = RUNTIME_DIR / "docs_upsert_batch_status.json"
DOCS_BATCH_HISTORY = RUNTIME_DIR / "docs_upsert_batch_history.jsonl"
PROOF_REFRESH_STATUS = RUNTIME_DIR / "proof_refresh_status.json"
PROOF_REFRESH_HISTORY = RUNTIME_DIR / "proof_refresh_history.jsonl"
RUNTIME_BENCHMARK_RUN_STATUS = RUNTIME_DIR / "runtime_benchmark_run_status.json"
RUNTIME_BENCHMARK_RUN_HISTORY = RUNTIME_DIR / "runtime_benchmark_run_history.jsonl"
RUNTIME_ARTIFACT_AUDIT_STATUS = RUNTIME_DIR / "runtime_artifact_audit_status.json"
RUNTIME_ARTIFACT_AUDIT_HISTORY = RUNTIME_DIR / "runtime_artifact_audit_history.jsonl"
CODE_FORGE_PROVENANCE_AUDIT_STATUS = RUNTIME_DIR / "code_forge_provenance_audit_status.json"
CODE_FORGE_PROVENANCE_AUDIT_HISTORY = RUNTIME_DIR / "code_forge_provenance_audit_history.jsonl"
CODE_FORGE_ARCHIVE_PLAN_STATUS = RUNTIME_DIR / "code_forge_archive_plan_status.json"
CODE_FORGE_ARCHIVE_PLAN_HISTORY = RUNTIME_DIR / "code_forge_archive_plan_history.jsonl"
CODE_FORGE_ARCHIVE_LIFECYCLE_STATUS = RUNTIME_DIR / "code_forge_archive_lifecycle_status.json"
CODE_FORGE_ARCHIVE_LIFECYCLE_HISTORY = RUNTIME_DIR / "code_forge_archive_lifecycle_history.jsonl"
FILE_FORGE_INDEX_STATUS = RUNTIME_DIR / "file_forge_index_status.json"
FILE_FORGE_INDEX_HISTORY = RUNTIME_DIR / "file_forge_index_history.jsonl"
FILE_FORGE_DB = FORGE_ROOT / "data" / "file_forge" / "library.sqlite"
SESSION_BRIDGE_DIR = RUNTIME_DIR / "session_bridge"
SESSION_BRIDGE_CONTEXT = SESSION_BRIDGE_DIR / "latest_context.json"
SESSION_BRIDGE_IMPORT_STATUS = SESSION_BRIDGE_DIR / "import_status.json"
PROOF_REPORT_DIR = FORGE_ROOT / "reports" / "proof"
PROOF_BUNDLE_DIR = FORGE_ROOT / "reports" / "proof_bundle"
SECURITY_REPORT_DIR = FORGE_ROOT / "reports" / "security"
RUNTIME_ARTIFACT_REPORT_DIR = FORGE_ROOT / "reports" / "runtime_artifact_audit"
CODE_FORGE_PROVENANCE_REPORT_DIR = FORGE_ROOT / "reports" / "code_forge_provenance_audit"
CODE_FORGE_ARCHIVE_PLAN_REPORT_DIR = FORGE_ROOT / "reports" / "code_forge_archive_plan"
CODE_FORGE_ARCHIVE_LIFECYCLE_REPORT_DIR = FORGE_ROOT / "reports" / "code_forge_archive_lifecycle"
CODE_FORGE_ARCHIVE_RETIREMENTS_LATEST = FORGE_ROOT / "data" / "code_forge" / "archive_ingestion" / "latest" / "retirements" / "latest.json"
SERVICES_SCRIPT = FORGE_ROOT / "scripts" / "eidos_termux_services.sh"
SERVICE_ACTION_LOG = RUNTIME_DIR / "atlas_service_actions.log"
SCHEDULER_CONTROL_SCRIPT = FORGE_ROOT / "scripts" / "eidos_scheduler_control.py"
TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eidos_dashboard")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_forge_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["EIDOS_FORGE_ROOT"] = str(FORGE_ROOT)
    existing = [p for p in str(env.get("PYTHONPATH") or "").split(os.pathsep) if p]
    merged: list[str] = []
    for path in [str(p) for p in FORGE_IMPORT_PATHS if p.exists()] + existing:
        if path and path not in merged:
            merged.append(path)
    env["PYTHONPATH"] = os.pathsep.join(merged)
    return env


SHELL_SESSIONS: Dict[str, Dict[str, Any]] = {}
SHELL_SESSIONS_LOCK = threading.Lock()


def _resolve_operator_path(raw_path: str = "", *, allow_home: bool = True) -> Path:
    candidate = (FORGE_ROOT / raw_path).resolve() if raw_path and not raw_path.startswith("/") else Path(raw_path or FORGE_ROOT).resolve()
    allowed_roots = [FORGE_ROOT.resolve()]
    if allow_home:
        allowed_roots.append(HOME_ROOT.resolve())
    for root in allowed_roots:
        try:
            candidate.relative_to(root)
            return candidate
        except Exception:
            continue
    raise HTTPException(status_code=403, detail="Path is outside allowed operator roots")


def _shell_binary() -> str:
    candidate = os.environ.get("SHELL", "").strip() or "/bin/sh"
    if Path(candidate).exists():
        return candidate
    return "/bin/sh" if Path("/bin/sh").exists() else candidate


def _set_pty_size(fd: int, cols: int, rows: int) -> None:
    cols = max(20, int(cols))
    rows = max(8, int(rows))
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def _spawn_shell_session(*, cwd: str = "", cols: int = 120, rows: int = 28) -> Dict[str, Any]:
    target_cwd = _resolve_operator_path(cwd or str(FORGE_ROOT), allow_home=True)
    master_fd, slave_fd = pty.openpty()
    _set_pty_size(master_fd, cols=cols, rows=rows)
    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    env = _build_forge_subprocess_env()
    env.setdefault("TERM", "xterm-256color")
    env.setdefault("HOME", str(HOME_ROOT))
    proc = subprocess.Popen(
        [_shell_binary()],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        cwd=str(target_cwd),
        env=env,
        start_new_session=True,
        close_fds=True,
    )
    os.close(slave_fd)
    session_id = f"shell:{uuid.uuid4().hex[:12]}"
    payload = {
        "contract": "eidos.atlas_shell_session.v1",
        "session_id": session_id,
        "shell": _shell_binary(),
        "cwd": str(target_cwd),
        "status": "running",
        "phase": "interactive",
        "started_at": _now_utc_iso(),
        "pid": proc.pid,
        "fd": master_fd,
        "process": proc,
    }
    with SHELL_SESSIONS_LOCK:
        SHELL_SESSIONS[session_id] = payload
    return _shell_session_payload(session_id)


def _shell_session_payload(session_id: str, *, include_output: bool = False, max_bytes: int = 16384) -> Dict[str, Any]:
    with SHELL_SESSIONS_LOCK:
        session = SHELL_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="shell session not found")
    proc: subprocess.Popen[Any] = session["process"]
    alive = proc.poll() is None
    if not alive:
        session["status"] = "exited"
        session["phase"] = "completed"
    payload = {
        "contract": "eidos.atlas_shell_session.v1",
        "session_id": session_id,
        "shell": session.get("shell"),
        "cwd": session.get("cwd"),
        "status": session.get("status", "running"),
        "phase": session.get("phase", "interactive"),
        "started_at": session.get("started_at"),
        "pid": session.get("pid"),
        "returncode": proc.poll(),
    }
    if include_output:
        chunks: list[str] = []
        remaining = max(256, int(max_bytes))
        while remaining > 0:
            try:
                ready, _, _ = select.select([session["fd"]], [], [], 0)
            except Exception:
                break
            if not ready:
                break
            try:
                data = os.read(session["fd"], min(4096, remaining))
            except BlockingIOError:
                break
            except OSError:
                break
            if not data:
                break
            chunks.append(data.decode("utf-8", errors="replace"))
            remaining -= len(data)
        payload["output"] = "".join(chunks)
    return payload


def _write_shell_input(session_id: str, text: str) -> Dict[str, Any]:
    with SHELL_SESSIONS_LOCK:
        session = SHELL_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="shell session not found")
    proc: subprocess.Popen[Any] = session["process"]
    if proc.poll() is not None:
        session["status"] = "exited"
        session["phase"] = "completed"
        return _shell_session_payload(session_id)
    os.write(session["fd"], text.encode("utf-8"))
    return _shell_session_payload(session_id)


def _resize_shell_session(session_id: str, *, cols: int, rows: int) -> Dict[str, Any]:
    with SHELL_SESSIONS_LOCK:
        session = SHELL_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="shell session not found")
    _set_pty_size(session["fd"], cols=cols, rows=rows)
    return _shell_session_payload(session_id)


def _stop_shell_session(session_id: str) -> Dict[str, Any]:
    with SHELL_SESSIONS_LOCK:
        session = SHELL_SESSIONS.pop(session_id, None)
    if not session:
        raise HTTPException(status_code=404, detail="shell session not found")
    proc: subprocess.Popen[Any] = session["process"]
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
    try:
        os.close(session["fd"])
    except Exception:
        pass
    session["status"] = "stopped"
    session["phase"] = "completed"
    return {
        "contract": "eidos.atlas_shell_session.v1",
        "session_id": session_id,
        "status": "stopped",
        "returncode": proc.poll(),
    }


def _shell_sessions_snapshot() -> Dict[str, Any]:
    entries = []
    with SHELL_SESSIONS_LOCK:
        session_ids = list(SHELL_SESSIONS.keys())
    for session_id in session_ids:
        entries.append(_shell_session_payload(session_id))
    return {"contract": "eidos.atlas_shell_snapshot.v1", "entries": entries}


def _cleanup_shell_sessions() -> None:
    with SHELL_SESSIONS_LOCK:
        session_ids = list(SHELL_SESSIONS.keys())
    for session_id in session_ids:
        try:
            _stop_shell_session(session_id)
        except Exception:
            continue


atexit.register(_cleanup_shell_sessions)

# --- App Setup ---
app = FastAPI(title="Eidosian Atlas", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# --- Helpers ---


def get_system_stats() -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    try:
        payload["cpu"] = psutil.cpu_percent(interval=None)
    except Exception as e:
        logger.warning(f"Unable to read CPU percent: {e}")
        payload["cpu"] = None
    try:
        mem = psutil.virtual_memory()
        payload.update(
            {
                "ram_percent": mem.percent,
                "ram_used_gb": round(mem.used / (1024**3), 2),
                "ram_total_gb": round(mem.total / (1024**3), 2),
            }
        )
    except Exception as e:
        logger.warning(f"Unable to read memory stats: {e}")
    try:
        disk = psutil.disk_usage(str(FORGE_ROOT))
        payload.update(
            {
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2),
            }
        )
    except Exception as e:
        logger.warning(f"Unable to read disk stats: {e}")
    try:
        payload["uptime"] = int(psutil.boot_time())
    except Exception as e:
        logger.warning(f"Unable to read boot time: {e}")
    return payload


def _read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else default
    except Exception:
        return default


def get_forge_status() -> Dict[str, Any]:
    status = {"doc_forge": "unknown", "details": {}}
    if DOC_STATUS.exists():
        try:
            data = json.loads(DOC_STATUS.read_text())
            status["doc_forge"] = data.get("status", "unknown")
            status["details"] = data
        except Exception:
            pass
    return status


def get_doc_snapshot() -> Dict[str, Any]:
    status_payload = _read_json(DOC_STATUS, {})
    index_payload = _read_json(DOC_INDEX, {})
    entries = index_payload.get("entries", [])
    if not isinstance(entries, list):
        entries = []

    recent_docs = [entry for entry in entries if isinstance(entry, dict)]
    recent_docs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    recent_docs = recent_docs[:12]

    return {
        "status": status_payload,
        "index_count": len(entries),
        "recent_docs": recent_docs,
    }


def get_file_forge_index_status() -> Dict[str, Any]:
    return _read_json(FILE_FORGE_INDEX_STATUS, {"contract": "eidos.file_forge.index.status.v1", "status": "idle"})


def get_file_forge_index_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(FILE_FORGE_INDEX_HISTORY, limit=limit)


def get_file_forge_summary(path_prefix: str = "", recent_limit: int = 8) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "contract": "eidos.file_forge.summary.v1",
        "status": "missing",
        "db_path": str(FILE_FORGE_DB),
        "path_prefix": path_prefix or None,
        "latest_index": get_file_forge_index_status(),
        "recent_files": [],
        "by_kind": [],
        "by_forge": [],
    }
    if not FILE_FORGE_DB.exists():
        return payload
    summary = FileLibraryDB(FILE_FORGE_DB).summary(
        path_prefix=_resolve_operator_path(path_prefix, allow_home=True) if path_prefix else None,
        recent_limit=recent_limit,
    )
    for row in summary.get("recent_files", []):
        file_path = Path(str(row.get("file_path") or ""))
        try:
            row["file_path"] = str(file_path.relative_to(FORGE_ROOT))
        except Exception:
            row["file_path"] = str(file_path)
    payload.update(summary)
    payload["status"] = "ready"
    payload["db_exists"] = True
    return payload


def _docs_inventory(
    path_prefix: str = "",
    refresh: bool = False,
    missing_only: bool = False,
    suppressed_only: bool = False,
    review_only: bool = False,
    limit: int = 50,
) -> Dict[str, Any]:
    from doc_forge.scribe.directory_docs import inventory_summary, write_inventory_status  # type: ignore

    selected = {path_prefix.strip("/")} if path_prefix else None
    if (
        not refresh
        and not path_prefix
        and not missing_only
        and not suppressed_only
        and not review_only
        and DIRECTORY_DOCS_STATUS.exists()
    ):
        status_payload = _read_json(DIRECTORY_DOCS_STATUS, {})
        if status_payload:
            result = dict(status_payload)
            result.setdefault("records", [])
            result["returned_count"] = 0
            return result

    if not path_prefix and not missing_only and not suppressed_only and not review_only:
        status_payload = write_inventory_status(FORGE_ROOT, DIRECTORY_DOCS_STATUS, selected_paths=selected)
    else:
        status_payload = {}
    payload = inventory_summary(FORGE_ROOT, selected_paths=selected)
    records = [row for row in payload.get("records", []) if isinstance(row, dict)]
    if missing_only:
        records = [row for row in records if not row.get("has_readme")]
    if suppressed_only:
        records = [row for row in records if row.get("suppressed")]
    if review_only:
        records = [row for row in records if row.get("review_required")]
    payload["records"] = records[: max(1, int(limit))]
    payload["returned_count"] = len(payload["records"])
    if status_payload:
        payload["status"] = status_payload
    return payload


def _docs_tree(path_prefix: str = "", limit: int = 250, refresh: bool = False) -> Dict[str, Any]:
    from doc_forge.scribe.directory_docs import inventory_tree  # type: ignore

    if not refresh and not path_prefix:
        cached = _read_json(DIRECTORY_DOCS_TREE, {})
        if cached:
            return cached
        status = _read_json(DIRECTORY_DOCS_STATUS, {})
        missing_examples = [row for row in (status.get("missing_examples") or []) if isinstance(row, str)]
        if missing_examples:
            nodes = [
                {
                    "path": row,
                    "name": Path(row).name,
                    "parent_path": "" if Path(row).parent.as_posix() == "." else Path(row).parent.as_posix(),
                    "depth": len(Path(row).parts),
                    "has_readme": False,
                    "tracked_files": 0,
                    "tests_present": False,
                    "api_route_count": 0,
                    "child_directory_count": 0,
                    "summary": "Cached missing README example from scheduler status.",
                }
                for row in missing_examples[: max(1, int(limit))]
            ]
            return {
                "contract": "eidos.documentation_tree.v1",
                "generated_at": status.get("generated_at", ""),
                "repo_root": str(FORGE_ROOT),
                "selected_paths": [],
                "returned_count": len(nodes),
                "groups": [],
                "nodes": nodes,
            }
    selected = {path_prefix.strip("/")} if path_prefix else None
    payload = inventory_tree(FORGE_ROOT, selected_paths=selected, limit=limit)
    if not path_prefix:
        _write_json(DIRECTORY_DOCS_TREE, payload)
    return payload


def get_docs_history(limit: int = 60) -> List[Dict[str, Any]]:
    payload = _read_json(DIRECTORY_DOCS_HISTORY, {})
    rows = payload.get("entries", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)][-max(1, int(limit)) :]


def get_runtime_snapshot() -> Dict[str, Any]:
    coordinator = _read_json(COORDINATOR_STATUS, {})
    scheduler = _read_json(SCHEDULER_STATUS, {})
    local_agent = _read_json(LOCAL_AGENT_STATUS, {})
    qwenchat = _read_json(QWENCHAT_STATUS, {})
    living_pipeline = _read_json(LIVING_PIPELINE_STATUS, {})
    doc_processor = _read_json(DOC_STATUS, {})
    file_forge = get_file_forge_summary(recent_limit=6)
    boot_status = _read_json(BOOT_STATUS, {})
    capabilities = _read_json(CAPABILITIES_STATUS, {})
    directory_docs = _read_json(DIRECTORY_DOCS_STATUS, {})
    session_bridge = get_session_bridge_status()
    proof_summary = get_proof_summary()
    archive_plan = get_code_forge_archive_plan_status()
    archive_lifecycle = get_code_forge_archive_lifecycle_status()
    if not capabilities:
        capabilities = asdict(collect_runtime_capabilities())
    return {
        "coordinator": coordinator,
        "scheduler": scheduler,
        "local_agent": local_agent,
        "qwenchat": qwenchat,
        "living_pipeline": living_pipeline,
        "doc_processor": doc_processor,
        "file_forge": file_forge,
        "file_forge_index": get_file_forge_index_status(),
        "file_forge_index_history": get_file_forge_index_history(),
        "shell": _shell_sessions_snapshot(),
        "archive_plan": archive_plan,
        "archive_lifecycle": archive_lifecycle,
        "archive_plan_history": get_code_forge_archive_plan_history(),
        "archive_lifecycle_history": get_code_forge_archive_lifecycle_history(),
        "archive_plan_report": get_latest_code_forge_archive_plan(),
        "archive_lifecycle_report": get_latest_code_forge_archive_lifecycle(),
        "archive_lifecycle_retirements": get_latest_code_forge_archive_retirements(),
        "boot": boot_status,
        "capabilities": capabilities,
        "directory_docs": directory_docs,
        "directory_docs_history": get_docs_history(limit=12),
        "session_bridge": session_bridge,
        "proof": proof_summary.get("proof", {}),
        "proof_bundle": proof_summary.get("bundle", {}),
        "identity_continuity": proof_summary.get("identity", {}),
        "identity_history": proof_summary.get("identity_history", []),
        "proof_history": proof_summary.get("proof_history", []),
        "external_benchmarks": proof_summary.get("external_benchmarks", []),
        "runtime_benchmarks": proof_summary.get("runtime_benchmarks", []),
        "proof_refresh": get_proof_refresh_status(),
        "proof_refresh_history": get_proof_refresh_history(),
        "runtime_benchmark_run": get_runtime_benchmark_run_status(),
        "runtime_benchmark_run_history": get_runtime_benchmark_run_history(),
        "docs_batch": get_docs_batch_status(),
        "docs_batch_history": get_docs_batch_history(),
        "runtime_artifact_audit": get_runtime_artifact_audit_status(),
        "runtime_artifact_audit_history": get_runtime_artifact_audit_history(),
        "code_forge_provenance_audit": get_code_forge_provenance_audit_status(),
        "code_forge_provenance_audit_history": get_code_forge_provenance_audit_history(),
        "security": (proof_summary.get("security") or {}).get("summary", {}),
        "security_plan": (proof_summary.get("security") or {}).get("plan", {}),
        "proof_summary": proof_summary,
    }


def get_local_agent_history(limit: int = 12) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not LOCAL_AGENT_HISTORY.exists():
        return rows
    try:
        lines = LOCAL_AGENT_HISTORY.read_text(encoding="utf-8").splitlines()
    except Exception:
        return rows
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
        if len(rows) >= max(1, int(limit)):
            break
    return rows


def _read_jsonl_rows(path: Path, limit: int = 12) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return rows
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
        if len(rows) >= max(1, int(limit)):
            break
    return rows


def get_qwenchat_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(QWENCHAT_HISTORY, limit=limit)


def get_living_pipeline_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(LIVING_PIPELINE_HISTORY, limit=limit)


def get_doc_processor_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(DOC_HISTORY, limit=limit)


def get_scheduler_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(SCHEDULER_HISTORY, limit=limit)


def get_runtime_services_snapshot() -> List[Dict[str, Any]]:
    def _row(name: str, payload: Dict[str, Any], path: Path) -> Dict[str, Any]:
        return {
            "service": name,
            "status": payload.get("status") or payload.get("state"),
            "phase": payload.get("phase") or payload.get("current_task"),
            "path": str(path.relative_to(FORGE_ROOT)) if path.exists() else str(path),
        }

    rows = [
        _row("scheduler", _read_json(SCHEDULER_STATUS, {}), SCHEDULER_STATUS),
        _row("doc_processor", _read_json(DOC_STATUS, {}), DOC_STATUS),
        {
            "service": "file_forge",
            "status": get_file_forge_summary().get("status"),
            "phase": get_file_forge_index_status().get("status"),
            "path": str(FILE_FORGE_DB.relative_to(FORGE_ROOT)) if FILE_FORGE_DB.exists() else str(FILE_FORGE_DB),
        },
        _row("local_agent", _read_json(LOCAL_AGENT_STATUS, {}), LOCAL_AGENT_STATUS),
        _row("qwenchat", _read_json(QWENCHAT_STATUS, {}), QWENCHAT_STATUS),
        _row("living_pipeline", _read_json(LIVING_PIPELINE_STATUS, {}), LIVING_PIPELINE_STATUS),
        _row("archive_plan", _read_json(CODE_FORGE_ARCHIVE_PLAN_STATUS, {}), CODE_FORGE_ARCHIVE_PLAN_STATUS),
        _row("archive_lifecycle", _read_json(CODE_FORGE_ARCHIVE_LIFECYCLE_STATUS, {}), CODE_FORGE_ARCHIVE_LIFECYCLE_STATUS),
    ]
    shell_entries = _shell_sessions_snapshot().get("entries", [])
    rows.append({
        "service": "atlas_shell",
        "status": "running" if shell_entries else "idle",
        "phase": shell_entries[0].get("phase") if shell_entries else "ready",
        "path": str(FORGE_ROOT),
    })
    return rows


def get_runtime_history(limit: int = 24) -> List[Dict[str, Any]]:
    payload = _read_json(COORDINATOR_HISTORY, {})
    rows = payload.get("entries", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)][-max(1, int(limit)) :]


def get_latest_proof_report() -> Dict[str, Any]:
    latest = PROOF_REPORT_DIR / "entity_proof_scorecard_latest.json"
    payload = _read_json(latest, {})
    if not isinstance(payload, dict):
        return {}
    return payload


def get_proof_history(limit: int = 12) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(PROOF_REPORT_DIR.glob("entity_proof_scorecard_*.json"), reverse=True):
        if path.name.endswith("_latest.json"):
            continue
        payload = _read_json(path, {})
        if not payload:
            continue
        overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
        freshness = payload.get("freshness") if isinstance(payload.get("freshness"), dict) else {}
        regression = payload.get("regression") if isinstance(payload.get("regression"), dict) else {}
        rows.append(
            {
                "generated_at": payload.get("generated_at") or "",
                "overall_score": overall.get("score"),
                "status": overall.get("status", ""),
                "freshness_status": freshness.get("status", ""),
                "regression_status": regression.get("status", ""),
                "path": str(path.relative_to(FORGE_ROOT)),
            }
        )
        if len(rows) >= max(1, int(limit)):
            break
    return list(reversed(rows))


def get_external_benchmark_results(limit: int = 12) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = FORGE_ROOT / "reports" / "external_benchmarks"
    if not root.exists():
        return rows
    for latest in sorted(root.glob("*/latest.json")):
        payload = _read_json(latest, {})
        if not payload:
            continue
        rows.append(
            {
                "suite": payload.get("suite") or latest.parent.name,
                "score": payload.get("score"),
                "status": payload.get("status", ""),
                "participant": payload.get("participant", ""),
                "execution_mode": payload.get("execution_mode", ""),
                "generated_at": payload.get("generated_at", ""),
                "path": str(latest.relative_to(FORGE_ROOT)),
            }
        )
    rows.sort(key=lambda row: str(row.get("generated_at") or ""), reverse=True)
    return rows[: max(1, int(limit))]


def get_runtime_benchmark_statuses(limit: int = 12) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = FORGE_ROOT / "data" / "runtime" / "external_benchmarks" / "agencybench"
    if not root.exists():
        return rows
    for status_path in sorted(root.glob("**/status.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        payload = _read_json(status_path, {})
        if not payload:
            continue
        rows.append(
            {
                "scenario": payload.get("scenario") or status_path.parent.name,
                "engine": payload.get("engine", ""),
                "model": payload.get("model", ""),
                "status": payload.get("status", ""),
                "stop_reason": payload.get("stop_reason", ""),
                "completed_count": payload.get("completed_count", 0),
                "attempt_count": payload.get("attempt_count", 0),
                "generated_at": payload.get("generated_at", ""),
                "path": str(status_path.relative_to(FORGE_ROOT)),
                "run_root": payload.get("run_root", ""),
            }
        )
        if len(rows) >= max(1, int(limit)):
            break
    return rows


def get_latest_proof_bundle_manifest() -> Dict[str, Any]:
    latest = PROOF_BUNDLE_DIR / "latest_manifest.json"
    payload = _read_json(latest, {})
    if not isinstance(payload, dict):
        return {}
    return payload


def get_latest_identity_continuity_scorecard() -> Dict[str, Any]:
    latest = PROOF_REPORT_DIR / "identity_continuity_scorecard_latest.json"
    payload = _read_json(latest, {})
    if not isinstance(payload, dict):
        return {}
    return payload


def get_latest_dependabot_summary() -> Dict[str, Any]:
    for path in sorted(SECURITY_REPORT_DIR.glob("dependabot_open_summary_*.json"), reverse=True):
        payload = _read_json(path, {})
        if payload:
            payload["_path"] = str(path.relative_to(FORGE_ROOT))
            return payload
    return {}


def get_latest_dependabot_plan() -> Dict[str, Any]:
    for path in sorted(SECURITY_REPORT_DIR.glob("dependabot_remediation_plan_*.json"), reverse=True):
        payload = _read_json(path, {})
        if payload:
            payload["_path"] = str(path.relative_to(FORGE_ROOT))
            return payload
    return {}


def get_proof_summary() -> Dict[str, Any]:
    proof = get_latest_proof_report()
    bundle = get_latest_proof_bundle_manifest()
    identity = get_latest_identity_continuity_scorecard()
    identity_history = get_identity_continuity_history(limit=12)
    proof_history = get_proof_history(limit=12)
    external = get_external_benchmark_results(limit=12)
    runtime_benchmarks = get_runtime_benchmark_statuses(limit=12)
    session_bridge = get_session_bridge_status()
    security = get_latest_dependabot_summary()
    security_plan = get_latest_dependabot_plan()
    history = identity.get("history") if isinstance(identity.get("history"), dict) else {}
    return {
        "contract": "eidos.proof.summary.v1",
        "proof": proof,
        "bundle": bundle,
        "identity": identity,
        "identity_history": identity_history,
        "proof_history": proof_history,
        "external_benchmarks": external,
        "runtime_benchmarks": runtime_benchmarks,
        "security": {
            "summary": security,
            "plan": security_plan,
        },
        "session_bridge": {
            "recent_sessions": len(session_bridge.get("recent_sessions") or [])
            if isinstance(session_bridge.get("recent_sessions"), list)
            else 0,
            "last_sync_at": ((session_bridge.get("summary") or {}).get("last_sync_at")),
            "codex_records": ((session_bridge.get("summary") or {}).get("codex_records", 0)),
            "gemini_records": ((session_bridge.get("summary") or {}).get("gemini_records", 0)),
            "imported_records": ((session_bridge.get("summary") or {}).get("imported_records", 0)),
        },
        "identity_trend": history.get("trend"),
        "identity_delta": history.get("delta_from_previous"),
    }


def get_identity_continuity_history(limit: int = 12) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(PROOF_REPORT_DIR.glob("identity_continuity_scorecard_*.json"), reverse=True):
        if path.name.endswith("_latest.json"):
            continue
        payload = _read_json(path, {})
        if not isinstance(payload, dict) or not payload:
            continue
        rows.append(
            {
                "generated_at": payload.get("generated_at") or payload.get("ts") or "",
                "overall_score": payload.get("overall_score"),
                "status": payload.get("status", ""),
                "recent_sessions": ((payload.get("session_bridge") or {}).get("recent_sessions", 0)),
                "path": str(path.relative_to(FORGE_ROOT)),
            }
        )
        if len(rows) >= max(1, int(limit)):
            break
    return list(reversed(rows))


def _write_docs_batch_status(payload: Dict[str, Any]) -> None:
    DOCS_BATCH_STATUS.parent.mkdir(parents=True, exist_ok=True)
    DOCS_BATCH_STATUS.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def get_docs_batch_status() -> Dict[str, Any]:
    return _read_json(DOCS_BATCH_STATUS, {"contract": "eidos.docs_upsert_batch.status.v1", "status": "idle"})


def get_docs_batch_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(DOCS_BATCH_HISTORY, limit=limit)


def _write_job_status(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_job_history(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def get_proof_refresh_status() -> Dict[str, Any]:
    return _read_json(PROOF_REFRESH_STATUS, {"contract": "eidos.proof_refresh.status.v1", "status": "idle"})


def get_proof_refresh_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(PROOF_REFRESH_HISTORY, limit=limit)


def get_runtime_benchmark_run_status() -> Dict[str, Any]:
    return _read_json(
        RUNTIME_BENCHMARK_RUN_STATUS,
        {"contract": "eidos.runtime_benchmark_run.status.v1", "status": "idle"},
    )


def get_runtime_benchmark_run_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(RUNTIME_BENCHMARK_RUN_HISTORY, limit=limit)


def get_runtime_artifact_audit_status() -> Dict[str, Any]:
    return _read_json(
        RUNTIME_ARTIFACT_AUDIT_STATUS,
        {"contract": "eidos.runtime_artifact_audit.status.v1", "status": "idle"},
    )


def get_runtime_artifact_audit_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(RUNTIME_ARTIFACT_AUDIT_HISTORY, limit=limit)


def get_code_forge_provenance_audit_status() -> Dict[str, Any]:
    return _read_json(
        CODE_FORGE_PROVENANCE_AUDIT_STATUS,
        {"contract": "eidos.code_forge_provenance_audit.status.v1", "status": "idle"},
    )


def get_code_forge_provenance_audit_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(CODE_FORGE_PROVENANCE_AUDIT_HISTORY, limit=limit)


def get_code_forge_archive_plan_status() -> Dict[str, Any]:
    return _read_json(
        CODE_FORGE_ARCHIVE_PLAN_STATUS,
        {"contract": "eidos.code_forge_archive_plan.status.v1", "status": "idle"},
    )


def get_code_forge_archive_plan_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(CODE_FORGE_ARCHIVE_PLAN_HISTORY, limit=limit)


def get_latest_code_forge_archive_plan() -> Dict[str, Any]:
    return _read_json(CODE_FORGE_ARCHIVE_PLAN_REPORT_DIR / "latest.json", {})


def get_code_forge_archive_lifecycle_status() -> Dict[str, Any]:
    return _read_json(
        CODE_FORGE_ARCHIVE_LIFECYCLE_STATUS,
        {"contract": "eidos.code_forge_archive_lifecycle.status.v1", "status": "idle"},
    )


def get_code_forge_archive_lifecycle_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(CODE_FORGE_ARCHIVE_LIFECYCLE_HISTORY, limit=limit)


def get_latest_code_forge_archive_lifecycle() -> Dict[str, Any]:
    return _read_json(CODE_FORGE_ARCHIVE_LIFECYCLE_REPORT_DIR / "latest.json", {})


def get_latest_code_forge_archive_retirements() -> Dict[str, Any]:
    payload = _read_json(CODE_FORGE_ARCHIVE_RETIREMENTS_LATEST, {})
    repo_root = str(payload.get("repo_root") or "")
    if repo_root and Path(repo_root).resolve() != FORGE_ROOT.resolve():
        return {}
    return payload


def get_session_bridge_status() -> Dict[str, Any]:
    payload = {
        "contract": "eidos.session_bridge.status.v1",
        "context": _read_json(SESSION_BRIDGE_CONTEXT, {}),
        "import_status": _read_json(SESSION_BRIDGE_IMPORT_STATUS, {}),
    }
    try:
        from eidosian_runtime.session_bridge import recent_session_digest, summarize_import_status  # type: ignore

        payload["recent_sessions"] = recent_session_digest(limit=6)
        payload["summary"] = summarize_import_status(payload.get("import_status"))
    except Exception as exc:
        payload["recent_sessions"] = []
        payload["summary"] = {}
        payload["error"] = str(exc)
    return payload


def _run_docs_upsert_batch_job(*, limit: int, missing_only: bool, path_prefix: str, dry_run: bool) -> None:
    from doc_forge.scribe.directory_docs import upsert_directory_batch  # type: ignore

    started_at = _now_utc_iso()
    running_payload = {
        "contract": "eidos.docs_upsert_batch.status.v1",
        "status": "running",
        "started_at": started_at,
        "limit": int(limit),
        "missing_only": bool(missing_only),
        "path_prefix": path_prefix,
        "dry_run": bool(dry_run),
    }
    _write_docs_batch_status(running_payload)
    _append_job_history(DOCS_BATCH_HISTORY, running_payload)
    try:
        result = upsert_directory_batch(
            FORGE_ROOT,
            path_prefix=path_prefix,
            missing_only=missing_only,
            limit=limit,
            dry_run=dry_run,
        )
        final_payload = {
            "contract": "eidos.docs_upsert_batch.status.v1",
            "status": "completed",
            "started_at": started_at,
            "finished_at": _now_utc_iso(),
            "limit": int(limit),
            "missing_only": bool(missing_only),
            "path_prefix": path_prefix,
            "dry_run": bool(dry_run),
            "result": result,
        }
        _write_docs_batch_status(final_payload)
        _append_job_history(DOCS_BATCH_HISTORY, final_payload)
    except Exception as exc:
        error_payload = {
            "contract": "eidos.docs_upsert_batch.status.v1",
            "status": "error",
            "started_at": started_at,
            "finished_at": _now_utc_iso(),
            "limit": int(limit),
            "missing_only": bool(missing_only),
            "path_prefix": path_prefix,
            "dry_run": bool(dry_run),
            "error": str(exc),
        }
        _write_docs_batch_status(error_payload)
        _append_job_history(DOCS_BATCH_HISTORY, error_payload)


def _run_runtime_artifact_audit_job(*, policy_path: str = "") -> None:
    started_at = _now_utc_iso()
    running_payload = {
        "contract": "eidos.runtime_artifact_audit.status.v1",
        "status": "running",
        "started_at": started_at,
        "policy_path": policy_path,
    }
    _write_job_status(RUNTIME_ARTIFACT_AUDIT_STATUS, running_payload)
    _append_job_history(RUNTIME_ARTIFACT_AUDIT_HISTORY, running_payload)
    try:
        from eidosian_runtime.artifact_policy import (  # type: ignore
            audit_runtime_artifacts,
            write_runtime_artifact_audit,
            write_runtime_artifact_audit_markdown,
        )

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report = audit_runtime_artifacts(FORGE_ROOT, policy_path=policy_path or None)
        report["generated_at"] = _now_utc_iso()
        report["policy_override"] = policy_path or None
        report_json_path = RUNTIME_ARTIFACT_REPORT_DIR / f"runtime_artifact_audit_{stamp}.json"
        report_md_path = RUNTIME_ARTIFACT_REPORT_DIR / f"runtime_artifact_audit_{stamp}.md"
        latest_json_path = RUNTIME_ARTIFACT_REPORT_DIR / "latest.json"
        latest_md_path = RUNTIME_ARTIFACT_REPORT_DIR / "latest.md"
        write_runtime_artifact_audit(FORGE_ROOT, report_json_path, policy_path=policy_path or None)
        latest_json_path.parent.mkdir(parents=True, exist_ok=True)
        latest_json_path.write_text(report_json_path.read_text(encoding="utf-8"), encoding="utf-8")
        write_runtime_artifact_audit_markdown(report, report_md_path)
        latest_md_path.write_text(report_md_path.read_text(encoding="utf-8"), encoding="utf-8")
        final_payload = {
            "contract": "eidos.runtime_artifact_audit.status.v1",
            "status": "completed",
            "started_at": started_at,
            "finished_at": _now_utc_iso(),
            "policy_path": policy_path,
            "tracked_violation_count": report.get("tracked_violation_count"),
            "live_generated_count": report.get("live_generated_count"),
            "latest_report": str(report_json_path.relative_to(FORGE_ROOT)),
            "latest_markdown": str(report_md_path.relative_to(FORGE_ROOT)),
        }
        _write_job_status(RUNTIME_ARTIFACT_AUDIT_STATUS, final_payload)
        _append_job_history(RUNTIME_ARTIFACT_AUDIT_HISTORY, final_payload)
    except Exception as exc:
        error_payload = {
            "contract": "eidos.runtime_artifact_audit.status.v1",
            "status": "error",
            "started_at": started_at,
            "finished_at": _now_utc_iso(),
            "policy_path": policy_path,
            "error": str(exc),
        }
        _write_job_status(RUNTIME_ARTIFACT_AUDIT_STATUS, error_payload)
        _append_job_history(RUNTIME_ARTIFACT_AUDIT_HISTORY, error_payload)


def _run_file_forge_index_job(*, target_path: str, remove_after_ingest: bool = False, max_files: int = 0) -> None:
    started_at = _now_utc_iso()
    running_payload = {
        "contract": "eidos.file_forge.index.status.v1",
        "status": "running",
        "started_at": started_at,
        "target_path": target_path,
        "remove_after_ingest": bool(remove_after_ingest),
        "max_files": int(max_files),
    }
    _write_job_status(FILE_FORGE_INDEX_STATUS, running_payload)
    _append_job_history(FILE_FORGE_INDEX_HISTORY, running_payload)
    try:
        resolved = _resolve_operator_path(target_path or str(FORGE_ROOT), allow_home=True)
        if not resolved.exists() or not resolved.is_dir():
            raise FileNotFoundError(f"directory not found: {resolved}")
        forge = FileForge(base_path=FORGE_ROOT)
        result = forge.index_directory(
            resolved,
            db_path=FILE_FORGE_DB,
            remove_after_ingest=remove_after_ingest,
            max_files=max_files or None,
        )
        final_payload = {
            "contract": "eidos.file_forge.index.status.v1",
            "status": "completed",
            "started_at": started_at,
            "finished_at": _now_utc_iso(),
            "target_path": str(resolved),
            "remove_after_ingest": bool(remove_after_ingest),
            "max_files": int(max_files),
            "indexed": result.get("indexed", 0),
            "skipped": result.get("skipped", 0),
            "removed": result.get("removed", 0),
            "db_path": result.get("db_path", str(FILE_FORGE_DB)),
        }
        _write_job_status(FILE_FORGE_INDEX_STATUS, final_payload)
        _append_job_history(FILE_FORGE_INDEX_HISTORY, final_payload)
    except Exception as exc:
        error_payload = {
            "contract": "eidos.file_forge.index.status.v1",
            "status": "error",
            "started_at": started_at,
            "finished_at": _now_utc_iso(),
            "target_path": target_path,
            "remove_after_ingest": bool(remove_after_ingest),
            "max_files": int(max_files),
            "error": str(exc),
        }
        _write_job_status(FILE_FORGE_INDEX_STATUS, error_payload)
        _append_job_history(FILE_FORGE_INDEX_HISTORY, error_payload)


def _run_code_forge_provenance_audit_job(*, limit: int = 12) -> None:
    started_at = _now_utc_iso()
    running_payload = {
        "contract": "eidos.code_forge_provenance_audit.status.v1",
        "status": "running",
        "started_at": started_at,
        "limit": int(limit),
    }
    _write_job_status(CODE_FORGE_PROVENANCE_AUDIT_STATUS, running_payload)
    _append_job_history(CODE_FORGE_PROVENANCE_AUDIT_HISTORY, running_payload)
    env = _build_forge_subprocess_env()
    python_bin = str(FORGE_ROOT / "eidosian_venv" / "bin" / "python")
    try:
        result = subprocess.run(
            [
                python_bin,
                str(FORGE_ROOT / "scripts" / "code_forge_provenance_audit.py"),
                "--repo-root",
                str(FORGE_ROOT),
                "--limit",
                str(max(1, int(limit))),
            ],
            cwd=str(FORGE_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        latest_report = _read_json(CODE_FORGE_PROVENANCE_REPORT_DIR / "latest.json", {})
        final_payload = {
            "contract": "eidos.code_forge_provenance_audit.status.v1",
            "status": "completed" if result.returncode == 0 else "error",
            "started_at": started_at,
            "finished_at": _now_utc_iso(),
            "limit": int(limit),
            "returncode": result.returncode,
            "link_file_count": latest_report.get("link_file_count"),
            "registry_file_count": latest_report.get("registry_file_count"),
            "invalid_file_count": latest_report.get("invalid_file_count"),
            "latest_report": str((CODE_FORGE_PROVENANCE_REPORT_DIR / "latest.json").relative_to(FORGE_ROOT)),
            "latest_markdown": str((CODE_FORGE_PROVENANCE_REPORT_DIR / "latest.md").relative_to(FORGE_ROOT)),
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        }
        _write_job_status(CODE_FORGE_PROVENANCE_AUDIT_STATUS, final_payload)
        _append_job_history(CODE_FORGE_PROVENANCE_AUDIT_HISTORY, final_payload)
    except Exception as exc:
        error_payload = {
            "contract": "eidos.code_forge_provenance_audit.status.v1",
            "status": "error",
            "started_at": started_at,
            "finished_at": _now_utc_iso(),
            "limit": int(limit),
            "error": str(exc),
        }
        _write_job_status(CODE_FORGE_PROVENANCE_AUDIT_STATUS, error_payload)
        _append_job_history(CODE_FORGE_PROVENANCE_AUDIT_HISTORY, error_payload)


def _run_code_forge_archive_plan_job(*, refresh: bool = True) -> None:
    env = _build_forge_subprocess_env()
    python_bin = str(FORGE_ROOT / "eidosian_venv" / "bin" / "python")
    command = [
        python_bin,
        str(FORGE_ROOT / "scripts" / "code_forge_archive_plan.py"),
        "--repo-root",
        str(FORGE_ROOT),
    ]
    if refresh:
        command.append("--refresh")
    subprocess.run(
        command,
        cwd=str(FORGE_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )


def _run_code_forge_archive_lifecycle_status_job(*, refresh: bool = False, repo_keys: List[str] | None = None) -> None:
    env = _build_forge_subprocess_env()
    python_bin = str(FORGE_ROOT / "eidosian_venv" / "bin" / "python")
    command = [
        python_bin,
        str(FORGE_ROOT / "scripts" / "code_forge_archive_lifecycle.py"),
        "status",
        "--repo-root",
        str(FORGE_ROOT),
    ]
    for repo_key in repo_keys or []:
        command.extend(["--repo-key", repo_key])
    if refresh:
        command.append("--refresh")
    subprocess.run(
        command,
        cwd=str(FORGE_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )


def _run_code_forge_archive_wave_job(*, repo_keys: List[str] | None = None, batch_limit: int | None = None, refresh: bool = False, retry_failed: bool = False) -> None:
    env = _build_forge_subprocess_env()
    python_bin = str(FORGE_ROOT / "eidosian_venv" / "bin" / "python")
    command = [
        python_bin,
        str(FORGE_ROOT / "scripts" / "code_forge_archive_lifecycle.py"),
        "run-wave",
        "--repo-root",
        str(FORGE_ROOT),
    ]
    for repo_key in repo_keys or []:
        command.extend(["--repo-key", repo_key])
    if batch_limit is not None:
        command.extend(["--batch-limit", str(max(1, int(batch_limit)))])
    if refresh:
        command.append("--refresh")
    if retry_failed:
        command.append("--retry-failed")
    subprocess.run(
        command,
        cwd=str(FORGE_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )


def _run_code_forge_archive_lifecycle_cli(*args: str, timeout: int = 3600) -> Dict[str, Any]:
    env = _build_forge_subprocess_env()
    python_bin = str(FORGE_ROOT / "eidosian_venv" / "bin" / "python")
    result = subprocess.run(
        [python_bin, str(FORGE_ROOT / "scripts" / "code_forge_archive_lifecycle.py"), *args],
        cwd=str(FORGE_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    payload: Dict[str, Any] = {}
    try:
        payload = json.loads(result.stdout or "{}")
    except Exception:
        payload = {}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "payload": payload,
        "stderr": result.stderr[-4000:],
    }


def _run_code_forge_archive_preview_job(*, repo_keys: List[str] | None = None, refresh: bool = False, assume_remove_mode: bool = False) -> None:
    env = _build_forge_subprocess_env()
    python_bin = str(FORGE_ROOT / "eidosian_venv" / "bin" / "python")
    command = [
        python_bin,
        str(FORGE_ROOT / "scripts" / "code_forge_archive_lifecycle.py"),
        "preview-retire",
        "--repo-root",
        str(FORGE_ROOT),
    ]
    for repo_key in repo_keys or []:
        command.extend(["--repo-key", repo_key])
    if refresh:
        command.append("--refresh")
    if assume_remove_mode:
        command.append("--assume-remove-mode")
    subprocess.run(
        command,
        cwd=str(FORGE_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )


def _run_code_forge_archive_retire_job(*, repo_keys: List[str] | None = None, refresh: bool = False, dry_run: bool = True) -> None:
    env = _build_forge_subprocess_env()
    python_bin = str(FORGE_ROOT / "eidosian_venv" / "bin" / "python")
    command = [
        python_bin,
        str(FORGE_ROOT / "scripts" / "code_forge_archive_lifecycle.py"),
        "retire",
        "--repo-root",
        str(FORGE_ROOT),
    ]
    for repo_key in repo_keys or []:
        command.extend(["--repo-key", repo_key])
    if refresh:
        command.append("--refresh")
    if dry_run:
        command.append("--dry-run")
    subprocess.run(
        command,
        cwd=str(FORGE_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )


def _run_code_forge_archive_restore_job(*, repo_key: str) -> None:
    env = _build_forge_subprocess_env()
    python_bin = str(FORGE_ROOT / "eidosian_venv" / "bin" / "python")
    subprocess.run(
        [
            python_bin,
            str(FORGE_ROOT / "scripts" / "code_forge_archive_lifecycle.py"),
            "restore",
            "--repo-root",
            str(FORGE_ROOT),
            "--repo-key",
            repo_key,
        ],
        cwd=str(FORGE_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )


def _run_code_forge_archive_prune_job(*, repo_keys: List[str] | None = None, dry_run: bool = False) -> None:
    env = _build_forge_subprocess_env()
    python_bin = str(FORGE_ROOT / "eidosian_venv" / "bin" / "python")
    command = [
        python_bin,
        str(FORGE_ROOT / "scripts" / "code_forge_archive_lifecycle.py"),
        "prune-retired",
        "--repo-root",
        str(FORGE_ROOT),
    ]
    for repo_key in repo_keys or []:
        command.extend(["--repo-key", repo_key])
    if dry_run:
        command.append("--dry-run")
    subprocess.run(
        command,
        cwd=str(FORGE_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )


def _run_proof_refresh_job(*, window_days: int) -> None:
    started_at = _now_utc_iso()
    running_payload = {
        "contract": "eidos.proof_refresh.status.v1",
        "status": "running",
        "started_at": started_at,
        "window_days": int(window_days),
    }
    _write_job_status(PROOF_REFRESH_STATUS, running_payload)
    _append_job_history(PROOF_REFRESH_HISTORY, running_payload)
    env = _build_forge_subprocess_env()
    python_bin = str(FORGE_ROOT / "eidosian_venv" / "bin" / "python")
    try:
        proof = subprocess.run(
            [
                python_bin,
                str(FORGE_ROOT / "scripts" / "entity_proof_suite.py"),
                "--repo-root",
                str(FORGE_ROOT),
                "--window-days",
                str(max(1, int(window_days))),
            ],
            cwd=str(FORGE_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        bundle = subprocess.run(
            [
                python_bin,
                str(FORGE_ROOT / "scripts" / "export_entity_proof_bundle.py"),
                "--repo-root",
                str(FORGE_ROOT),
            ],
            cwd=str(FORGE_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        final_payload = {
            "contract": "eidos.proof_refresh.status.v1",
            "status": "completed" if proof.returncode == 0 and bundle.returncode == 0 else "error",
            "started_at": started_at,
            "finished_at": _now_utc_iso(),
            "window_days": int(window_days),
            "proof_returncode": proof.returncode,
            "bundle_returncode": bundle.returncode,
            "proof_stdout": proof.stdout[-4000:],
            "proof_stderr": proof.stderr[-4000:],
            "bundle_stdout": bundle.stdout[-4000:],
            "bundle_stderr": bundle.stderr[-4000:],
            "latest_proof": get_latest_proof_report(),
            "latest_bundle": get_latest_proof_bundle_manifest(),
        }
        _write_job_status(PROOF_REFRESH_STATUS, final_payload)
        _append_job_history(PROOF_REFRESH_HISTORY, final_payload)
    except Exception as exc:
        error_payload = {
            "contract": "eidos.proof_refresh.status.v1",
            "status": "error",
            "started_at": started_at,
            "finished_at": _now_utc_iso(),
            "window_days": int(window_days),
            "error": str(exc),
        }
        _write_job_status(PROOF_REFRESH_STATUS, error_payload)
        _append_job_history(PROOF_REFRESH_HISTORY, error_payload)


def _run_runtime_benchmark_job(
    *,
    scenario: str,
    engine: str,
    model: str,
    attempts_per_step: int,
    timeout_sec: float,
    keep_alive: str,
) -> None:
    started_at = _now_utc_iso()
    running_payload = {
        "contract": "eidos.runtime_benchmark_run.status.v1",
        "status": "running",
        "started_at": started_at,
        "scenario": scenario,
        "engine": engine,
        "model": model,
        "attempts_per_step": int(attempts_per_step),
        "timeout_sec": float(timeout_sec),
        "keep_alive": keep_alive,
    }
    _write_job_status(RUNTIME_BENCHMARK_RUN_STATUS, running_payload)
    _append_job_history(RUNTIME_BENCHMARK_RUN_HISTORY, running_payload)
    env = _build_forge_subprocess_env()
    python_bin = str(FORGE_ROOT / "eidosian_venv" / "bin" / "python")
    try:
        result = subprocess.run(
            [
                python_bin,
                str(FORGE_ROOT / "scripts" / "run_agencybench_eidos.py"),
                "--scenario",
                scenario,
                "--engine",
                engine,
                "--model",
                model,
                "--attempts-per-step",
                str(max(1, int(attempts_per_step))),
                "--timeout-sec",
                str(max(60.0, float(timeout_sec))),
                "--keep-alive",
                keep_alive,
            ],
            cwd=str(FORGE_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(120.0, float(timeout_sec)) + 300.0,
            check=False,
        )
        payload: Dict[str, Any] = {}
        try:
            payload = json.loads(result.stdout) if result.stdout.strip() else {}
        except Exception:
            payload = {}
        final_payload = {
            "contract": "eidos.runtime_benchmark_run.status.v1",
            "status": "completed" if result.returncode == 0 else "error",
            "started_at": started_at,
            "finished_at": _now_utc_iso(),
            "scenario": scenario,
            "engine": engine,
            "model": model,
            "attempts_per_step": int(attempts_per_step),
            "timeout_sec": float(timeout_sec),
            "keep_alive": keep_alive,
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "result": payload if isinstance(payload, dict) else {},
        }
        _write_job_status(RUNTIME_BENCHMARK_RUN_STATUS, final_payload)
        _append_job_history(RUNTIME_BENCHMARK_RUN_HISTORY, final_payload)
    except Exception as exc:
        error_payload = {
            "contract": "eidos.runtime_benchmark_run.status.v1",
            "status": "error",
            "started_at": started_at,
            "finished_at": _now_utc_iso(),
            "scenario": scenario,
            "engine": engine,
            "model": model,
            "attempts_per_step": int(attempts_per_step),
            "timeout_sec": float(timeout_sec),
            "keep_alive": keep_alive,
            "error": str(exc),
        }
        _write_job_status(RUNTIME_BENCHMARK_RUN_STATUS, error_payload)
        _append_job_history(RUNTIME_BENCHMARK_RUN_HISTORY, error_payload)


def get_file_tree(path: Path, root: Path) -> List[Dict[str, Any]]:
    tree = []
    try:
        # Sort directories first, then files
        entries = sorted(list(path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
        for entry in entries:
            if entry.name.startswith(".") or entry.name == "__pycache__":
                continue
            item = {
                "name": entry.name,
                "path": str(entry.relative_to(root)),
                "is_dir": entry.is_dir(),
                "size": entry.stat().st_size if entry.is_file() else 0,
                "mtime": datetime.fromtimestamp(entry.stat().st_mtime).isoformat(),
            }
            if entry.is_dir():
                item["children"] = []  # Lazy loading not implemented for simplicity, just show dir
            tree.append(item)
    except Exception as e:
        logger.error(f"Error listing {path}: {e}")
    return tree


def _resolve_browse_root(domain: str) -> Path:
    if domain == "forge":
        return FORGE_ROOT
    if domain == "home":
        return HOME_ROOT
    return DOC_FINAL


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
    allowed = {"start", "stop", "pause", "resume", "restart", "status", "low-load", "restore-standard"}
    if action not in allowed:
        raise HTTPException(status_code=400, detail="Invalid service action")
    if not SERVICES_SCRIPT.exists():
        raise HTTPException(status_code=503, detail="Service controller unavailable")
    allowed_services = {
        None,
        "",
        "all",
        "ollama-qwen",
        "ollama-embedding",
        "mcp",
        "doc-forge",
        "atlas",
        "scheduler",
        "local-agent",
    }
    service = (service or "").strip() or None
    if service not in allowed_services:
        raise HTTPException(status_code=400, detail="Invalid service target")
    env = _build_forge_subprocess_env()
    if action != "status":
        await to_thread(_run_service_action_async, action, service)
        return {
            "action": action,
            "service": service or "all",
            "accepted": True,
            "queued": True,
            "ok": True,
            "services": [],
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


def _scheduler_command(action: str) -> Dict[str, Any]:
    allowed = {"status", "pause", "resume", "stop"}
    if action not in allowed:
        raise HTTPException(status_code=400, detail="Invalid scheduler action")
    if not SCHEDULER_CONTROL_SCRIPT.exists():
        raise HTTPException(status_code=503, detail="Scheduler controller unavailable")
    env = _build_forge_subprocess_env()
    result = subprocess.run(
        [
            str(FORGE_ROOT / "eidosian_venv" / "bin" / "python"),
            str(SCHEDULER_CONTROL_SCRIPT),
            action,
        ],
        cwd=str(FORGE_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(result.stdout or "{}")
    except Exception:
        payload = {}
    return {
        "action": action,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "ok": result.returncode == 0,
        "payload": payload if isinstance(payload, dict) else {},
    }


def _parse_service_status_output(raw: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or ":" not in line or line.startswith("Interactive shell refcount"):
            continue
        name, state = line.split(":", 1)
        state_value = state.strip()
        rows.append(
            {
                "name": name.strip(),
                "state": state_value,
                "running": ("run:" in state_value or "running(" in state_value) and "paused" not in state_value,
                "paused": "paused" in state_value,
            }
        )
    return rows


# --- Routes ---


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    sys_stats = get_system_stats()
    forge_status = get_forge_status()
    doc_snapshot = get_doc_snapshot()
    docs_inventory = _docs_inventory(limit=12)
    docs_reviews = _docs_inventory(limit=12, review_only=True)
    docs_suppressed = _docs_inventory(limit=12, suppressed_only=True)
    docs_tree = _docs_tree(limit=40, refresh=False)
    docs_history = get_docs_history(limit=24)
    runtime_snapshot = get_runtime_snapshot()
    runtime_services = get_runtime_services_snapshot()
    code_forge_provenance_audit_history = get_code_forge_provenance_audit_history()
    local_agent_history = get_local_agent_history()
    scheduler_history = get_scheduler_history()
    doc_processor_history = get_doc_processor_history()
    qwenchat_history = get_qwenchat_history()
    living_pipeline_history = get_living_pipeline_history()
    proof_snapshot = get_latest_proof_report()
    proof_summary = get_proof_summary()

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "sys_stats": sys_stats,
            "forge_status": forge_status,
            "recent_docs": doc_snapshot["recent_docs"],
            "doc_status": doc_snapshot["status"],
            "doc_index_count": doc_snapshot["index_count"],
            "docs_inventory": docs_inventory,
            "docs_reviews": docs_reviews,
            "docs_suppressed": docs_suppressed,
            "docs_tree": docs_tree,
            "docs_history": docs_history,
            "runtime_snapshot": runtime_snapshot,
            "runtime_services": runtime_services,
            "code_forge_provenance_audit_history": code_forge_provenance_audit_history,
            "local_agent_history": local_agent_history,
            "scheduler_history": scheduler_history,
            "doc_processor_history": doc_processor_history,
            "qwenchat_history": qwenchat_history,
            "living_pipeline_history": living_pipeline_history,
            "proof_snapshot": proof_snapshot,
            "proof_summary": proof_summary,
            "service_snapshot": await _service_command("status"),
        },
    )


@app.get("/browse/{domain}/", response_class=HTMLResponse)
async def browse_domain_root(request: Request, domain: str):
    return await browse_domain(request, domain, ".")


@app.get("/browse/{domain}/{path:path}", response_class=HTMLResponse)
async def browse_domain(request: Request, domain: str, path: str = "."):
    root = _resolve_browse_root(domain)
    target_path = (root / path).resolve()
    if not str(target_path).startswith(str(DOC_FINAL.resolve())):
        if not str(target_path).startswith(str(root.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")

    if domain == "docs" and not str(target_path).startswith(str(DOC_FINAL.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    if target_path.is_dir():
        files = get_file_tree(target_path, root)
        parent = str(Path(path).parent) if path != "." else None
        return templates.TemplateResponse(
            request,
            "browser.html",
            {
                "request": request,
                "domain": domain,
                "path": path,
                "files": files,
                "parent": parent,
                "root_label": root.name,
            },
        )
    elif target_path.is_file():
        if target_path.suffix.lower() == ".md":
            content = target_path.read_text()
            html_content = markdown.markdown(content, extensions=["fenced_code", "tables", "toc"])
            return templates.TemplateResponse(
                request,
                "viewer.html",
                {"request": request, "path": path, "content": html_content, "filename": target_path.name},
            )
        else:
            # For non-markdown files, just serve raw or download
            return FileResponse(target_path)


@app.get("/browse/{path:path}", response_class=HTMLResponse)
async def browse(request: Request, path: str):
    return await browse_domain(request, "docs", path)


@app.get("/api/system")
async def api_system():
    return get_system_stats()


@app.get("/api/doc/status")
async def api_doc_status():
    return get_doc_snapshot()


@app.get("/api/docs/coverage")
async def api_docs_coverage(
    limit: int = 50,
    missing_only: bool = False,
    suppressed_only: bool = False,
    review_only: bool = False,
    path_prefix: str = "",
    refresh: bool = False,
):
    return _docs_inventory(
        path_prefix=path_prefix,
        refresh=refresh,
        missing_only=missing_only,
        suppressed_only=suppressed_only,
        review_only=review_only,
        limit=limit,
    )


@app.get("/api/docs/tree")
async def api_docs_tree(limit: int = 250, path_prefix: str = "", refresh: bool = False):
    return _docs_tree(path_prefix=path_prefix, limit=limit, refresh=refresh)


@app.get("/api/docs/render")
async def api_docs_render(path: str):
    from doc_forge.scribe.directory_docs import render_directory_readme  # type: ignore

    return {"path": path, "content": render_directory_readme(FORGE_ROOT, path)}


@app.get("/api/docs/readme")
async def api_docs_readme(path: str):
    target = (FORGE_ROOT / path / "README.md").resolve()
    if not str(target).startswith(str(FORGE_ROOT.resolve())) or not target.exists():
        raise HTTPException(status_code=404, detail="README not found")
    return {"path": path, "content": target.read_text(encoding="utf-8")}


@app.get("/api/docs/diff")
async def api_docs_diff(path: str):
    from doc_forge.scribe.directory_docs import readme_diff  # type: ignore

    return {"path": path, "diff": readme_diff(FORGE_ROOT, path)}


@app.post("/api/docs/upsert")
async def api_docs_upsert(path: str):
    from doc_forge.scribe.directory_docs import upsert_directory_readme  # type: ignore

    return upsert_directory_readme(FORGE_ROOT, path)


@app.post("/api/docs/upsert-batch")
async def api_docs_upsert_batch(
    limit: int = 50,
    missing_only: bool = True,
    path_prefix: str = "",
    dry_run: bool = False,
    background: bool = False,
):
    from doc_forge.scribe.directory_docs import upsert_directory_batch  # type: ignore

    if background:
        thread = threading.Thread(
            target=_run_docs_upsert_batch_job,
            kwargs={
                "limit": limit,
                "missing_only": missing_only,
                "path_prefix": path_prefix,
                "dry_run": dry_run,
            },
            daemon=True,
        )
        thread.start()
        return {
            "contract": "eidos.docs_upsert_batch.status.v1",
            "status": "queued",
            "limit": int(limit),
            "missing_only": bool(missing_only),
            "path_prefix": path_prefix,
            "dry_run": bool(dry_run),
        }

    return upsert_directory_batch(
        FORGE_ROOT,
        path_prefix=path_prefix,
        missing_only=missing_only,
        limit=limit,
        dry_run=dry_run,
    )


@app.get("/api/docs/upsert-batch/status")
async def api_docs_upsert_batch_status():
    return get_docs_batch_status()


@app.get("/api/docs/upsert-batch/history")
async def api_docs_upsert_batch_history(limit: int = 12):
    return {
        "contract": "eidos.docs_upsert_batch.history.v1",
        "entries": get_docs_batch_history(limit=limit),
    }


@app.get("/api/session-bridge")
async def api_session_bridge():
    return get_session_bridge_status()


@app.post("/api/session-bridge/sync")
async def api_session_bridge_sync():
    from eidosian_runtime.session_bridge import sync_external_sessions  # type: ignore

    result = sync_external_sessions(min_interval_sec=0.0)
    payload = get_session_bridge_status()
    payload["sync_result"] = result
    return payload


@app.post("/api/docs/refresh")
async def api_docs_refresh():
    return _docs_inventory(refresh=True, limit=20)


@app.get("/api/docs/history")
async def api_docs_history(limit: int = 60):
    return {"contract": "eidos.documentation_history.v1", "entries": get_docs_history(limit=limit)}


@app.get("/api/runtime")
async def api_runtime():
    snapshot = get_runtime_snapshot()
    snapshot["history"] = get_runtime_history()
    return snapshot


@app.get("/api/runtime/services")
async def api_runtime_services():
    return {
        "contract": "eidos.runtime_services_snapshot.v1",
        "entries": get_runtime_services_snapshot(),
    }


@app.get("/api/proof/latest")
async def api_proof_latest():
    payload = get_latest_proof_report()
    if payload:
        return payload
    raise HTTPException(status_code=404, detail="No proof report found")


@app.get("/api/proof/summary")
async def api_proof_summary():
    payload = get_proof_summary()
    if payload.get("proof"):
        return payload
    raise HTTPException(status_code=404, detail="No proof summary found")


@app.get("/api/proof/history")
async def api_proof_history(limit: int = 12):
    return {
        "contract": "eidos.proof.history.v1",
        "entries": get_proof_history(limit=limit),
    }


@app.post("/api/proof/refresh")
async def api_proof_refresh(window_days: int = 30, background: bool = True):
    if background:
        thread = threading.Thread(
            target=_run_proof_refresh_job,
            kwargs={"window_days": window_days},
            daemon=True,
        )
        thread.start()
        return {
            "contract": "eidos.proof_refresh.status.v1",
            "status": "queued",
            "window_days": int(window_days),
        }
    _run_proof_refresh_job(window_days=window_days)
    return get_proof_refresh_status()


@app.get("/api/proof/refresh/status")
async def api_proof_refresh_status():
    return get_proof_refresh_status()


@app.get("/api/proof/refresh/history")
async def api_proof_refresh_history(limit: int = 12):
    return {
        "contract": "eidos.proof_refresh.history.v1",
        "entries": get_proof_refresh_history(limit=limit),
    }


@app.get("/api/proof/bundle/latest")
async def api_proof_bundle_latest():
    payload = get_latest_proof_bundle_manifest()
    if payload:
        return payload
    raise HTTPException(status_code=404, detail="No proof bundle manifest found")


@app.get("/api/proof/identity/latest")
async def api_proof_identity_latest():
    payload = get_latest_identity_continuity_scorecard()
    if payload:
        return payload
    raise HTTPException(status_code=404, detail="No identity continuity scorecard found")


@app.get("/api/proof/identity/history")
async def api_proof_identity_history(limit: int = 12):
    return {
        "contract": "eidos.identity_continuity_history.v1",
        "entries": get_identity_continuity_history(limit=limit),
    }


@app.get("/api/proof/external")
async def api_proof_external(limit: int = 12):
    return {
        "contract": "eidos.external_benchmark_snapshot.v1",
        "entries": get_external_benchmark_results(limit=limit),
    }


@app.get("/api/benchmarks/runtime")
async def api_runtime_benchmarks(limit: int = 12):
    return {
        "contract": "eidos.runtime_benchmark_snapshot.v1",
        "entries": get_runtime_benchmark_statuses(limit=limit),
    }


@app.post("/api/benchmarks/runtime/run")
async def api_runtime_benchmark_run(
    scenario: str = "scenario2",
    engine: str = "local_agent",
    model: str = "qwen3.5:2b",
    attempts_per_step: int = 1,
    timeout_sec: float = 900.0,
    keep_alive: str = "4h",
    background: bool = True,
):
    if scenario not in {"scenario1", "scenario2"}:
        raise HTTPException(status_code=400, detail="Invalid benchmark scenario")
    if engine not in {"local_agent", "deterministic"}:
        raise HTTPException(status_code=400, detail="Invalid benchmark engine")
    if background:
        thread = threading.Thread(
            target=_run_runtime_benchmark_job,
            kwargs={
                "scenario": scenario,
                "engine": engine,
                "model": model,
                "attempts_per_step": attempts_per_step,
                "timeout_sec": timeout_sec,
                "keep_alive": keep_alive,
            },
            daemon=True,
        )
        thread.start()
        return {
            "contract": "eidos.runtime_benchmark_run.status.v1",
            "status": "queued",
            "scenario": scenario,
            "engine": engine,
            "model": model,
        }
    _run_runtime_benchmark_job(
        scenario=scenario,
        engine=engine,
        model=model,
        attempts_per_step=attempts_per_step,
        timeout_sec=timeout_sec,
        keep_alive=keep_alive,
    )
    return get_runtime_benchmark_run_status()


@app.get("/api/benchmarks/runtime/run/status")
async def api_runtime_benchmark_run_status():
    return get_runtime_benchmark_run_status()


@app.get("/api/benchmarks/runtime/run/history")
async def api_runtime_benchmark_run_history(limit: int = 12):
    return {
        "contract": "eidos.runtime_benchmark_run.history.v1",
        "entries": get_runtime_benchmark_run_history(limit=limit),
    }


@app.post("/api/code-forge/provenance-audit")
async def api_code_forge_provenance_audit(limit: int = 12, background: bool = True):
    if background:
        thread = threading.Thread(
            target=_run_code_forge_provenance_audit_job,
            kwargs={"limit": limit},
            daemon=True,
        )
        thread.start()
        return {
            "contract": "eidos.code_forge_provenance_audit.status.v1",
            "status": "queued",
            "limit": int(limit),
        }

    _run_code_forge_provenance_audit_job(limit=limit)
    return get_code_forge_provenance_audit_status()


@app.get("/api/code-forge/provenance-audit/status")
async def api_code_forge_provenance_audit_status():
    return get_code_forge_provenance_audit_status()


@app.get("/api/code-forge/provenance-audit/history")
async def api_code_forge_provenance_audit_history(limit: int = 12):
    return {
        "contract": "eidos.code_forge_provenance_audit.history.v1",
        "entries": get_code_forge_provenance_audit_history(limit=limit),
    }


@app.get("/api/code-forge/archive-plan")
async def api_code_forge_archive_plan():
    return {
        "status": get_code_forge_archive_plan_status(),
        "history": get_code_forge_archive_plan_history(),
        "report": get_latest_code_forge_archive_plan(),
    }


@app.post("/api/code-forge/archive-plan")
async def api_code_forge_archive_plan_run(refresh: bool = True, background: bool = True):
    if background:
        thread = threading.Thread(
            target=_run_code_forge_archive_plan_job,
            kwargs={"refresh": refresh},
            daemon=True,
        )
        thread.start()
        return {
            "contract": "eidos.code_forge_archive_plan.status.v1",
            "status": "queued",
            "refresh": bool(refresh),
        }
    _run_code_forge_archive_plan_job(refresh=refresh)
    return get_code_forge_archive_plan_status()


@app.get("/api/code-forge/archive-lifecycle")
async def api_code_forge_archive_lifecycle(limit: int = 12):
    return {
        "status": get_code_forge_archive_lifecycle_status(),
        "history": get_code_forge_archive_lifecycle_history(limit=limit),
        "report": get_latest_code_forge_archive_lifecycle(),
        "retirements": get_latest_code_forge_archive_retirements(),
    }


@app.post("/api/code-forge/archive-lifecycle/status")
async def api_code_forge_archive_lifecycle_status_job(repo_key: str = "", refresh: bool = False, background: bool = True):
    repo_keys = [item.strip() for item in repo_key.split(",") if item.strip()]
    if background:
        thread = threading.Thread(
            target=_run_code_forge_archive_lifecycle_status_job,
            kwargs={"refresh": refresh, "repo_keys": repo_keys},
            daemon=True,
        )
        thread.start()
        return {
            "contract": "eidos.code_forge_archive_lifecycle.status.v1",
            "status": "queued",
            "phase": "status",
            "refresh": bool(refresh),
            "repo_keys": repo_keys,
        }
    _run_code_forge_archive_lifecycle_status_job(refresh=refresh, repo_keys=repo_keys)
    return get_code_forge_archive_lifecycle_status()


@app.post("/api/code-forge/archive-lifecycle/run-wave")
async def api_code_forge_archive_lifecycle_wave(repo_key: str = "", batch_limit: int = 20, refresh: bool = False, retry_failed: bool = False, background: bool = True):
    repo_keys = [item.strip() for item in repo_key.split(",") if item.strip()]
    if background:
        thread = threading.Thread(
            target=_run_code_forge_archive_wave_job,
            kwargs={"repo_keys": repo_keys, "batch_limit": batch_limit, "refresh": refresh, "retry_failed": retry_failed},
            daemon=True,
        )
        thread.start()
        return {
            "contract": "eidos.code_forge_archive_lifecycle.status.v1",
            "status": "queued",
            "phase": "run_wave",
            "refresh": bool(refresh),
            "repo_keys": repo_keys,
            "batch_limit": int(batch_limit),
            "retry_failed": bool(retry_failed),
        }
    _run_code_forge_archive_wave_job(repo_keys=repo_keys, batch_limit=batch_limit, refresh=refresh, retry_failed=retry_failed)
    return get_code_forge_archive_lifecycle_status()


@app.post("/api/code-forge/archive-lifecycle/set-mode")
async def api_code_forge_archive_lifecycle_set_mode(repo_key: str, mode: str, reason: str = ""):
    if mode not in {"ingest_and_keep", "ingest_and_remove"}:
        raise HTTPException(status_code=400, detail="Invalid lifecycle mode")
    return _run_code_forge_archive_lifecycle_cli(
        "set-mode",
        "--repo-root",
        str(FORGE_ROOT),
        "--repo-key",
        repo_key,
        "--mode",
        mode,
        "--reason",
        reason,
        timeout=300,
    )


@app.post("/api/code-forge/archive-lifecycle/preview-retire")
async def api_code_forge_archive_lifecycle_preview_retire(repo_key: str = "", refresh: bool = False, assume_remove_mode: bool = False, background: bool = True):
    repo_keys = [item.strip() for item in repo_key.split(",") if item.strip()]
    if background:
        thread = threading.Thread(
            target=_run_code_forge_archive_preview_job,
            kwargs={"repo_keys": repo_keys, "refresh": refresh, "assume_remove_mode": assume_remove_mode},
            daemon=True,
        )
        thread.start()
        return {
            "contract": "eidos.code_forge_archive_lifecycle.status.v1",
            "status": "queued",
            "phase": "preview_retire",
            "refresh": bool(refresh),
            "repo_keys": repo_keys,
            "assume_remove_mode": bool(assume_remove_mode),
        }
    args = ["preview-retire", "--repo-root", str(FORGE_ROOT)]
    for item in repo_keys:
        args.extend(["--repo-key", item])
    if refresh:
        args.append("--refresh")
    if assume_remove_mode:
        args.append("--assume-remove-mode")
    return _run_code_forge_archive_lifecycle_cli(*args)


@app.post("/api/code-forge/archive-lifecycle/retire")
async def api_code_forge_archive_lifecycle_retire(repo_key: str = "", refresh: bool = False, dry_run: bool = True, background: bool = True):
    repo_keys = [item.strip() for item in repo_key.split(",") if item.strip()]
    if background:
        thread = threading.Thread(
            target=_run_code_forge_archive_retire_job,
            kwargs={"repo_keys": repo_keys, "refresh": refresh, "dry_run": dry_run},
            daemon=True,
        )
        thread.start()
        return {
            "contract": "eidos.code_forge_archive_lifecycle.status.v1",
            "status": "queued",
            "phase": "retire",
            "refresh": bool(refresh),
            "repo_keys": repo_keys,
            "dry_run": bool(dry_run),
        }
    args = ["retire", "--repo-root", str(FORGE_ROOT)]
    for item in repo_keys:
        args.extend(["--repo-key", item])
    if refresh:
        args.append("--refresh")
    if dry_run:
        args.append("--dry-run")
    return _run_code_forge_archive_lifecycle_cli(*args)


@app.post("/api/code-forge/archive-lifecycle/prune-retired")
async def api_code_forge_archive_lifecycle_prune(repo_key: str = "", dry_run: bool = False, background: bool = True):
    repo_keys = [item.strip() for item in repo_key.split(",") if item.strip()]
    if background:
        thread = threading.Thread(
            target=_run_code_forge_archive_prune_job,
            kwargs={"repo_keys": repo_keys, "dry_run": dry_run},
            daemon=True,
        )
        thread.start()
        return {
            "contract": "eidos.code_forge_archive_lifecycle.status.v1",
            "status": "queued",
            "phase": "prune_retired",
            "repo_keys": repo_keys,
            "dry_run": bool(dry_run),
        }
    args = ["prune-retired", "--repo-root", str(FORGE_ROOT)]
    for item in repo_keys:
        args.extend(["--repo-key", item])
    if dry_run:
        args.append("--dry-run")
    return _run_code_forge_archive_lifecycle_cli(*args)


@app.post("/api/code-forge/archive-lifecycle/restore")
async def api_code_forge_archive_lifecycle_restore(repo_key: str, background: bool = True):
    repo_key = repo_key.strip()
    if not repo_key:
        raise HTTPException(status_code=400, detail="repo_key is required")
    if background:
        thread = threading.Thread(
            target=_run_code_forge_archive_restore_job,
            kwargs={"repo_key": repo_key},
            daemon=True,
        )
        thread.start()
        return {
            "contract": "eidos.code_forge_archive_lifecycle.status.v1",
            "status": "queued",
            "phase": "restore",
            "repo_keys": [repo_key],
        }
    return _run_code_forge_archive_lifecycle_cli(
        "restore",
        "--repo-root",
        str(FORGE_ROOT),
        "--repo-key",
        repo_key,
    )


@app.post("/api/runtime-artifacts/audit")
async def api_runtime_artifacts_audit(policy_path: str = "", background: bool = True):
    if background:
        thread = threading.Thread(
            target=_run_runtime_artifact_audit_job,
            kwargs={"policy_path": policy_path},
            daemon=True,
        )
        thread.start()
        return {
            "contract": "eidos.runtime_artifact_audit.status.v1",
            "status": "queued",
            "policy_path": policy_path,
        }

    _run_runtime_artifact_audit_job(policy_path=policy_path)
    return get_runtime_artifact_audit_status()


@app.get("/api/runtime-artifacts/audit/status")
async def api_runtime_artifacts_audit_status():
    return get_runtime_artifact_audit_status()


@app.get("/api/runtime-artifacts/audit/history")
async def api_runtime_artifacts_audit_history(limit: int = 12):
    return {
        "contract": "eidos.runtime_artifact_audit.history.v1",
        "entries": get_runtime_artifact_audit_history(limit=limit),
    }


@app.get("/api/security/dependabot")
async def api_security_dependabot():
    summary = get_latest_dependabot_summary()
    plan = get_latest_dependabot_plan()
    if summary:
        return {
            "contract": "eidos.security.dependabot_snapshot.v1",
            "summary": summary,
            "plan": plan,
        }
    raise HTTPException(status_code=404, detail="No Dependabot summary found")


@app.get("/api/services")
async def api_services(service: str = ""):
    return await _service_command("status", service)


@app.post("/api/services/{action}")
async def api_services_action(action: str, service: str = ""):
    return await _service_command(action, service)


@app.get("/api/capabilities")
async def api_capabilities():
    return get_runtime_snapshot().get("capabilities", {})


@app.get("/api/scheduler")
async def api_scheduler():
    return _scheduler_command("status")


@app.post("/api/scheduler/{action}")
async def api_scheduler_action(action: str):
    return _scheduler_command(action)


@app.get("/api/runtime/local-agent")
async def api_local_agent_status():
    return {
        "status": _read_json(LOCAL_AGENT_STATUS, {}),
        "history": get_local_agent_history(),
    }


@app.get("/api/runtime/scheduler")
async def api_runtime_scheduler_status():
    return {
        "status": _read_json(SCHEDULER_STATUS, {}),
        "history": get_scheduler_history(),
    }


@app.get("/api/runtime/doc-processor")
async def api_doc_processor_status():
    return {
        "status": _read_json(DOC_STATUS, {}),
        "history": get_doc_processor_history(),
    }


@app.get("/api/runtime/qwenchat")
async def api_qwenchat_status():
    return {
        "status": _read_json(QWENCHAT_STATUS, {}),
        "history": get_qwenchat_history(),
    }


@app.get("/api/runtime/living-pipeline")
async def api_living_pipeline_status():
    return {
        "status": _read_json(LIVING_PIPELINE_STATUS, {}),
        "history": get_living_pipeline_history(),
    }


@app.get("/api/runtime/file-forge")
async def api_file_forge_runtime_status(path_prefix: str = ""):
    return {
        "summary": get_file_forge_summary(path_prefix=path_prefix, recent_limit=12),
        "index_status": get_file_forge_index_status(),
        "index_history": get_file_forge_index_history(),
    }


@app.post("/api/file-forge/index")
async def api_file_forge_index(path: str = "", remove_after_ingest: bool = False, max_files: int = 0, background: bool = True):
    if background:
        thread = threading.Thread(
            target=_run_file_forge_index_job,
            kwargs={"target_path": path, "remove_after_ingest": remove_after_ingest, "max_files": max_files},
            daemon=True,
        )
        thread.start()
        return {
            "contract": "eidos.file_forge.index.status.v1",
            "status": "queued",
            "target_path": path,
            "remove_after_ingest": bool(remove_after_ingest),
            "max_files": int(max_files),
        }
    _run_file_forge_index_job(target_path=path, remove_after_ingest=remove_after_ingest, max_files=max_files)
    return get_file_forge_index_status()


@app.get("/api/file-forge/index/status")
async def api_file_forge_index_status():
    return get_file_forge_index_status()


@app.get("/api/file-forge/index/history")
async def api_file_forge_index_history(limit: int = 12):
    return {"contract": "eidos.file_forge.index.history.v1", "entries": get_file_forge_index_history(limit=limit)}


@app.post("/api/shell/start")
async def api_shell_start(cwd: str = "", cols: int = 120, rows: int = 28):
    return _spawn_shell_session(cwd=cwd, cols=cols, rows=rows)


@app.get("/api/shell/status")
async def api_shell_status(session_id: str = ""):
    if session_id.strip():
        return _shell_session_payload(session_id.strip())
    return _shell_sessions_snapshot()


@app.get("/api/shell/read")
async def api_shell_read(session_id: str, max_bytes: int = 16384):
    return _shell_session_payload(session_id, include_output=True, max_bytes=max_bytes)


@app.post("/api/shell/input")
async def api_shell_input(session_id: str, text: str):
    return _write_shell_input(session_id, text)


@app.post("/api/shell/resize")
async def api_shell_resize(session_id: str, cols: int = 120, rows: int = 28):
    return _resize_shell_session(session_id, cols=cols, rows=rows)


@app.post("/api/shell/stop")
async def api_shell_stop(session_id: str):
    return _stop_shell_session(session_id)


@app.get("/api/consciousness")
async def api_consciousness():
    """Retrieve the latest metrics from the Consciousness Kernel."""
    try:
        from agent_forge.consciousness.kernel import ConsciousnessKernel

        payload = ConsciousnessKernel.read_runtime_health(FORGE_ROOT / "state")
        payload["status"] = "ok"
        return payload
    except Exception as e:
        logger.error(f"Error getting consciousness health: {e}")
        return {"status": "error", "error": str(e)}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "eidos_atlas"}


if __name__ == "__main__":
    import uvicorn

    # Default port 8936 for dashboard (next to doc_forge 8930)
    port = int(os.environ.get("EIDOS_DASHBOARD_PORT", 8936))
    uvicorn.run(app, host="0.0.0.0", port=port)
