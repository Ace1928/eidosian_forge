import fcntl
import os
import pty
import select
import signal
import struct
import subprocess
import termios
import threading
import uuid
import logging
import atexit
from pathlib import Path
from typing import Any, Dict
from fastapi import HTTPException
from .config import FORGE_ROOT, HOME_ROOT
from .utils import _now_utc_iso

logger = logging.getLogger("eidos_dashboard")

SHELL_SESSIONS: Dict[str, Dict[str, Any]] = {}
SHELL_SESSIONS_LOCK = threading.Lock()

def _resolve_operator_path(raw_path: str = "", *, allow_home: bool = True) -> Path:
    from .utils import _resolve_operator_path as utils_resolve
    return utils_resolve(raw_path, FORGE_ROOT, HOME_ROOT, allow_home=allow_home)

def _build_forge_subprocess_env() -> Dict[str, str]:
    from .config import setup_pythonpath
    setup_pythonpath()
    env = os.environ.copy()
    env["EIDOS_FORGE_ROOT"] = str(FORGE_ROOT)
    return env

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
            except (BlockingIOError, OSError):
                break
            if not data:
                break
            chunks.append(data.decode("utf-8", errors="replace"))
            remaining -= len(data)
        payload["output"] = "".join(chunks)
    return payload

def _shell_sessions_snapshot() -> Dict[str, Any]:
    entries = []
    with SHELL_SESSIONS_LOCK:
        session_ids = list(SHELL_SESSIONS.keys())
    for session_id in session_ids:
        try:
            entries.append(_shell_session_payload(session_id))
        except Exception:
            continue
    return {"contract": "eidos.atlas_shell_snapshot.v1", "entries": entries}

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
    return {"status": "stopped", "session_id": session_id}

def _cleanup_shell_sessions() -> None:
    with SHELL_SESSIONS_LOCK:
        session_ids = list(SHELL_SESSIONS.keys())
    for session_id in session_ids:
        try:
            _stop_shell_session(session_id)
        except Exception:
            continue

atexit.register(_cleanup_shell_sessions)
