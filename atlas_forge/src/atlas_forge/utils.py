import json
import logging
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from fastapi import HTTPException, Request

logger = logging.getLogger("eidos_dashboard")

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else default
    except Exception:
        return default

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

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

def _detect_lan_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            ip = probe.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    return None

def _resolve_operator_path(raw_path: str, root_dir: Path, home_dir: Path, allow_home: bool = True) -> Path:
    candidate = (root_dir / raw_path).resolve() if raw_path and not raw_path.startswith("/") else Path(raw_path or root_dir).resolve()
    allowed_roots = [root_dir.resolve()]
    if allow_home:
        allowed_roots.append(home_dir.resolve())
    for root in allowed_roots:
        try:
            candidate.relative_to(root)
            return candidate
        except Exception:
            continue
    raise HTTPException(status_code=403, detail="Path is outside allowed operator roots")
