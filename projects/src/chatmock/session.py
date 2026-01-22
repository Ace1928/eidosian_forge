from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple


_LOCK = threading.Lock()
_MAX_ENTRIES = 10000

# Persistence Setup
_CACHE_DIR = Path.home() / ".chatmock"
_SESSION_FILE = _CACHE_DIR / "sessions.json"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _load_sessions() -> Tuple[Dict[str, str], List[str]]:
    if not _SESSION_FILE.exists():
        return {}, []
    try:
        data = json.loads(_SESSION_FILE.read_text(encoding="utf-8"))
        return data.get("map", {}), data.get("order", [])
    except Exception:
        return {}, []

def _save_sessions(mapping: Dict[str, str], order: List[str]) -> None:
    try:
        tmp = _SESSION_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"map": mapping, "order": order}, indent=2), encoding="utf-8")
        tmp.replace(_SESSION_FILE)
    except Exception:
        pass

_FINGERPRINT_TO_UUID, _ORDER = _load_sessions()


def _canonicalize_first_user_message(input_items: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """
    Extract the first stable user message from Responses input items. Good use for a fingerprint for prompt caching.
    """
    for item in input_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        role = item.get("role")
        if role != "user":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        norm_content = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "input_text":
                text = part.get("text") if isinstance(part.get("text"), str) else ""
                if text:
                    norm_content.append({"type": "input_text", "text": text})
            elif ptype == "input_image":
                url = part.get("image_url") if isinstance(part.get("image_url"), str) else None
                if url:
                    norm_content.append({"type": "input_image", "image_url": url})
        if norm_content:
            return {"type": "message", "role": "user", "content": norm_content}
    return None


def canonicalize_prefix(instructions: str | None, input_items: List[Dict[str, Any]]) -> str:
    prefix: Dict[str, Any] = {}
    if isinstance(instructions, str) and instructions.strip():
        prefix["instructions"] = instructions.strip()
    first_user = _canonicalize_first_user_message(input_items)
    if first_user is not None:
        prefix["first_user_message"] = first_user
    return json.dumps(prefix, sort_keys=True, separators=(",", ":"))


def _fingerprint(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _remember(fp: str, sid: str) -> None:
    if fp in _FINGERPRINT_TO_UUID:
        return
    _FINGERPRINT_TO_UUID[fp] = sid
    _ORDER.append(fp)
    if len(_ORDER) > _MAX_ENTRIES:
        oldest = _ORDER.pop(0)
        _FINGERPRINT_TO_UUID.pop(oldest, None)
    
    # Persist asynchronously or immediately? Immediate is safer for now.
    _save_sessions(_FINGERPRINT_TO_UUID, _ORDER)


def ensure_session_id(
    instructions: str | None,
    input_items: List[Dict[str, Any]],
    client_supplied: str | None = None,
) -> str:
    if isinstance(client_supplied, str) and client_supplied.strip():
        return client_supplied.strip()

    canon = canonicalize_prefix(instructions, input_items)
    fp = _fingerprint(canon)
    with _LOCK:
        if fp in _FINGERPRINT_TO_UUID:
            return _FINGERPRINT_TO_UUID[fp]
        sid = str(uuid.uuid4())
        _remember(fp, sid)
        return sid