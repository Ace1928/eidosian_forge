from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import List, Optional

from ..forge_loader import ensure_forge_import
from ..transactions import (
    begin_transaction,
    check_idempotency,
    find_latest_transaction_for_path,
    hash_paths,
    list_transactions,
    load_transaction,
    record_idempotency,
)

ensure_forge_import("diagnostics_forge")
ensure_forge_import("file_forge")

from diagnostics_forge import DiagnosticsForge
from file_forge import FileForge

from ..core import tool


diag = DiagnosticsForge(service_name="mcp_system")
file_forge = FileForge()

FORGE_DIR = Path(os.environ.get("EIDOS_FORGE_DIR", "/home/lloyd/eidosian_forge"))
_DEFAULT_ALLOWED = [
    Path("/home/lloyd"),
    Path("/tmp"),
    FORGE_DIR,
]
_ALLOWED_ROOTS = [
    Path(p.strip()).expanduser().resolve()
    for p in os.environ.get("EIDOS_ALLOWED_PATHS", "").split(",")
    if p.strip()
] or [p.resolve() for p in _DEFAULT_ALLOWED]

MAX_READ_BYTES = int(os.environ.get("EIDOS_MAX_READ_BYTES", str(5 * 1024 * 1024)))
_READ_ONLY_PREFIXES = (
    "ls",
    "pwd",
    "whoami",
    "date",
    "rg",
    "cat",
    "head",
    "tail",
    "sed -n",
    "stat",
    "wc",
    "python -c",
    "python3 -c",
)


def _resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (FORGE_DIR / candidate).resolve()
    return candidate.resolve()


def _is_allowed(path: Path) -> bool:
    for root in _ALLOWED_ROOTS:
        try:
            if path == root or path.is_relative_to(root):
                return True
        except AttributeError:
            try:
                path.relative_to(root)
                return True
            except ValueError:
                continue
    return False


def _is_read_only_command(command: str) -> bool:
    stripped = command.strip()
    return any(
        stripped == prefix or stripped.startswith(f"{prefix} ")
        for prefix in _READ_ONLY_PREFIXES
    )


def _run_verify_command(command: Optional[str], cwd: Optional[Path]) -> Optional[str]:
    if not command:
        return None
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
    )
    if result.returncode != 0:
        return result.stderr or result.stdout or "Verification command failed"
    return None


@tool(
    description="Get system details.",
    parameters={"type": "object", "properties": {}},
)
def system_info() -> str:
    """Get system details."""
    return json.dumps(
        {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        indent=2,
    )


@tool(
    description="Read a file from disk.",
    parameters={
        "type": "object",
        "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"],
    },
)
def file_read(file_path: str, max_bytes: int = MAX_READ_BYTES) -> str:
    """Read a file."""
    path = _resolve_path(file_path)
    if not _is_allowed(path):
        return "Error: Path not allowed"
    if not path.exists():
        return "Error: File not found"
    if path.is_dir():
        return "Error: Path is a directory"
    if path.stat().st_size > max_bytes:
        return f"Error: File exceeds max_bytes ({max_bytes})"
    return path.read_text(encoding="utf-8")


@tool(
    description="Write content to a file.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "content": {"type": "string"},
            "overwrite": {"type": "boolean"},
        },
        "required": ["file_path", "content"],
    },
)
def file_write(file_path: str, content: str, overwrite: bool = True) -> str:
    """Write a file."""
    path = _resolve_path(file_path)
    if not _is_allowed(path):
        return "Error: Path not allowed"
    if path.exists() and path.is_dir():
        return "Error: Path is a directory"
    if path.exists() and not overwrite:
        return "No-op: File already exists"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return "No-op: Content unchanged"
    txn = begin_transaction("file_write", [path])
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(content, encoding="utf-8")
        if path.read_text(encoding="utf-8") != content:
            txn.rollback("verification_failed: content_mismatch")
            return f"Error: Verification failed; rolled back ({txn.id})"
        txn.commit()
        return f"Committed file_write ({txn.id})"
    except Exception as exc:
        txn.rollback(f"exception: {exc}")
        return f"Error: {exc} (rolled back {txn.id})"


@tool(
    description="Create an empty file at the specified path.",
    parameters={
        "type": "object",
        "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"],
    },
)
def file_create(file_path: str) -> str:
    """Create an empty file."""
    path = _resolve_path(file_path)
    if not _is_allowed(path):
        return "Error: Path not allowed"
    if path.exists():
        return "No-op: Path already exists"
    txn = begin_transaction("file_create", [path])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        if not path.exists():
            txn.rollback("verification_failed: missing_file")
            return f"Error: Verification failed; rolled back ({txn.id})"
        txn.commit()
        return f"Committed file_create ({txn.id})"
    except Exception as exc:
        txn.rollback(f"exception: {exc}")
        return f"Error: {exc} (rolled back {txn.id})"


@tool(
    description="Delete a file or an empty directory.",
    parameters={
        "type": "object",
        "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"],
    },
)
def file_delete(file_path: str) -> str:
    """Delete a file or empty directory."""
    path = _resolve_path(file_path)
    if not _is_allowed(path):
        return "Error: Path not allowed"
    if not path.exists():
        return "No-op: Path not found"
    if path.is_dir():
        if any(path.iterdir()):
            return "Error: Directory is not empty"
        txn = begin_transaction("file_delete", [path])
        try:
            path.rmdir()
            if path.exists():
                txn.rollback("verification_failed: dir_exists")
                return f"Error: Verification failed; rolled back ({txn.id})"
            txn.commit()
            return f"Committed file_delete ({txn.id})"
        except Exception as exc:
            txn.rollback(f"exception: {exc}")
            return f"Error: {exc} (rolled back {txn.id})"
    txn = begin_transaction("file_delete", [path])
    try:
        path.unlink()
        if path.exists():
            txn.rollback("verification_failed: file_exists")
            return f"Error: Verification failed; rolled back ({txn.id})"
        txn.commit()
        return f"Committed file_delete ({txn.id})"
    except Exception as exc:
        txn.rollback(f"exception: {exc}")
        return f"Error: {exc} (rolled back {txn.id})"


@tool(
    name="file_restore",
    description="Restore a file or directory from the latest transaction or a specified transaction id.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "transaction_id": {"type": "string"},
        },
        "required": ["file_path"],
    },
)
def file_restore(file_path: str, transaction_id: Optional[str] = None) -> str:
    """Restore a file or directory from a transaction snapshot."""
    path = _resolve_path(file_path)
    if not _is_allowed(path):
        return "Error: Path not allowed"
    txn_id = transaction_id or find_latest_transaction_for_path(path)
    if not txn_id:
        return "Error: No transaction found for path"
    txn = load_transaction(txn_id)
    if not txn:
        return "Error: Transaction not found"
    txn.rollback("manual_restore")
    return f"Restored {file_path} from {txn_id}"


@tool(
    name="run_shell_command",
    description="Execute a shell command and return stdout, stderr, and exit code.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "cwd": {"type": "string"},
            "timeout_sec": {"type": "integer"},
            "safe_mode": {"type": "boolean"},
            "transaction_paths": {"type": "array", "items": {"type": "string"}},
            "verify_command": {"type": "string"},
            "idempotency_key": {"type": "string"},
        },
        "required": ["command"],
    },
)
def run_shell_command(
    command: str,
    cwd: Optional[str] = None,
    timeout_sec: Optional[int] = None,
    safe_mode: bool = True,
    transaction_paths: Optional[List[str]] = None,
    verify_command: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> str:
    """Execute shell command."""
    diag.log_event("INFO", "Shell Exec", command=command, cwd=cwd)
    targets: List[Path] = []
    if transaction_paths:
        for raw in transaction_paths:
            resolved = _resolve_path(raw)
            if not _is_allowed(resolved):
                return json.dumps({"error": "Path not allowed", "path": str(resolved)})
            targets.append(resolved)
    if safe_mode and not _is_read_only_command(command) and not targets:
        return json.dumps(
            {
                "error": "Unsafe command blocked. Provide transaction_paths or disable safe_mode.",
                "command": command,
            }
        )
    txn = None
    if targets:
        if idempotency_key:
            target_state = hash_paths(targets)
            if check_idempotency(idempotency_key, command, target_state):
                return json.dumps(
                    {
                        "command": command,
                        "stdout": "",
                        "stderr": "",
                        "exit_code": 0,
                        "status": "no-op",
                        "idempotency_key": idempotency_key,
                    }
                )
        txn = begin_transaction("run_shell_command", targets)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd or None,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        if txn:
            txn.rollback("timeout")
        payload = {
            "command": command,
            "cwd": cwd or "",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "Command timed out",
            "exit_code": -1,
            "transaction_id": txn.id if txn else None,
            "rolled_back": bool(txn),
        }
        return json.dumps(payload)
    verify_error = _run_verify_command(verify_command, Path(cwd) if cwd else None)
    if txn and (result.returncode != 0 or verify_error):
        txn.rollback(verify_error or f"exit_code={result.returncode}")
    elif txn:
        txn.commit()
        if idempotency_key:
            record_idempotency(idempotency_key, command, hash_paths(targets))
    payload = {
        "command": command,
        "cwd": cwd or "",
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.returncode,
        "transaction_id": txn.id if txn else None,
        "rolled_back": bool(txn and (result.returncode != 0 or verify_error)),
    }
    return json.dumps(payload)


@tool(
    name="run_tests",
    description="Execute a test command and return stdout, stderr, and exit code.",
    parameters={
        "type": "object",
        "properties": {
            "test_command": {"type": "string"},
            "test_path": {"type": "string"},
        },
        "required": ["test_command"],
    },
)
def run_tests(
    test_command: str,
    test_path: Optional[str] = None,
    timeout_sec: Optional[int] = None,
) -> str:
    """Execute test command."""
    command = test_command if not test_path else f"{test_command} {test_path}"
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        payload = {
            "command": command,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "Test command timed out",
            "exit_code": -1,
        }
        return json.dumps(payload)
    payload = {
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.returncode,
    }
    return json.dumps(payload)


@tool(
    name="venv_run",
    description="Execute a command inside a Python virtual environment.",
    parameters={
        "type": "object",
        "properties": {
            "venv_path": {"type": "string"},
            "command": {"type": "string"},
        },
        "required": ["venv_path", "command"],
    },
)
def venv_run(
    venv_path: str,
    command: str,
    timeout_sec: Optional[int] = None,
) -> str:
    """Execute a command within a virtual environment."""
    venv_dir = Path(venv_path).expanduser()
    python_path = venv_dir / "bin" / "python"
    if os.name == "nt":
        python_path = venv_dir / "Scripts" / "python.exe"

    if not python_path.exists():
        return json.dumps(
            {
                "command": command,
                "stdout": "",
                "stderr": "",
                "exit_code": 1,
                "error": "Python executable not found in virtual environment",
            }
        )

    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = f"{python_path.parent}{os.pathsep}{env.get('PATH', '')}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        payload = {
            "command": command,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "Command timed out",
            "exit_code": -1,
        }
        return json.dumps(payload)
    payload = {
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.returncode,
    }
    return json.dumps(payload)


@tool(
    name="transaction_list",
    description="List recent transactional snapshots.",
    parameters={
        "type": "object",
        "properties": {"limit": {"type": "integer"}},
    },
)
def transaction_list(limit: int = 50) -> str:
    """List recent transactional snapshots."""
    return json.dumps(list_transactions(limit=limit), indent=2)


@tool(
    name="transaction_restore",
    description="Restore a transaction snapshot by id.",
    parameters={
        "type": "object",
        "properties": {"transaction_id": {"type": "string"}},
        "required": ["transaction_id"],
    },
)
def transaction_restore(transaction_id: str) -> str:
    """Restore a transaction snapshot by id."""
    txn = load_transaction(transaction_id)
    if not txn:
        return "Error: Transaction not found"
    txn.rollback("manual_restore")
    return f"Restored transaction {transaction_id}"


@tool(
    name="file_search",
    description="Search file contents for a string pattern.",
)
def file_search(pattern: str, root_path: Optional[str] = None, max_results: int = 100) -> str:
    """Search file contents for a string pattern."""
    root = _resolve_path(root_path) if root_path else FORGE_DIR
    if not _is_allowed(root):
        return "Error: Path not allowed"
    matches = file_forge.search_content(pattern, directory=root)
    matches = matches[:max_results]
    return json.dumps([str(p) for p in matches], indent=2)


@tool(
    name="file_find_duplicates",
    description="Find duplicate files by content hash.",
)
def file_find_duplicates(
    root_path: Optional[str] = None, max_groups: int = 50
) -> str:
    """Find duplicate files by content hash."""
    root = _resolve_path(root_path) if root_path else FORGE_DIR
    if not _is_allowed(root):
        return "Error: Path not allowed"
    dupes = file_forge.find_duplicates(root)
    limited = list(dupes.items())[:max_groups]
    payload = {h: [str(p) for p in paths] for h, paths in limited}
    return json.dumps(payload, indent=2)
