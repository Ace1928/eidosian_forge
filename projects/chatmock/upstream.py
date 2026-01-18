from __future__ import annotations

import uuid
import os
import json
import hashlib
import requests
from typing import Any, Dict, List, Iterator
from pathlib import Path
from flask import Response, current_app, jsonify, make_response
from .config import CHATGPT_RESPONSES_URL
from .session import ensure_session_id
from flask import request as flask_request
from .utils import get_effective_chatgpt_auth


def _log_json(prefix: str, payload: Any) -> None:
    try:
        print(f"{prefix}\n{json.dumps(payload, indent=2, ensure_ascii=False)}")
    except Exception:
        try:
            print(f"{prefix}\n{payload}")
        except Exception:
            pass


# Caching Setup
_CACHE_DIR = Path.home() / ".chatmock" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _compute_payload_hash(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class CachedResponse:
    def __init__(self, byte_generator: Iterator[bytes], status_code: int = 200, headers: Dict[str, Any] = None):
        self._gen = byte_generator
        self.status_code = status_code
        self.headers = headers or {}
        self._content = None

    def iter_lines(self, chunk_size=512, decode_unicode=False, delimiter=None) -> Iterator[Any]:
        pending = None
        for chunk in self._gen:
            if pending:
                chunk = pending + chunk
                pending = None
            
            if delimiter:
                lines = chunk.split(delimiter)
            else:
                lines = chunk.split(b'\n')
            
            if chunk.endswith(b'\n'):
                lines.pop()
            else:
                pending = lines.pop()
            
            for line in lines:
                if decode_unicode:
                    yield line.decode("utf-8")
                else:
                    yield line
        
        if pending:
            if decode_unicode:
                yield pending.decode("utf-8")
            else:
                yield pending

    @property
    def content(self) -> bytes:
        if self._content is None:
            self._content = b"".join(self._gen)
        return self._content

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def close(self):
        pass


def _read_from_cache_gen(cache_key: str) -> Iterator[bytes] | None:
    path = _CACHE_DIR / f"{cache_key}.bin"
    if not path.exists():
        return None
    
    def _gen():
        with path.open("rb") as f:
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                yield chunk
    return _gen()


def _tee_stream(iterator: Iterator[bytes], cache_key: str) -> Iterator[bytes]:
    """Wraps an iterator to save its content to a cache file."""
    path = _CACHE_DIR / f"{cache_key}.bin"
    tmp_path = _CACHE_DIR / f"{cache_key}.tmp"
    
    try:
        f = tmp_path.open("wb")
    except Exception:
        yield from iterator
        return

    try:
        for chunk in iterator:
            f.write(chunk)
            yield chunk
    except Exception:
        f.close()
        if tmp_path.exists():
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise
    else:
        f.close()
        try:
            tmp_path.replace(path)
        except Exception:
            pass


def normalize_model_name(name: str | None, debug_model: str | None = None) -> str:
    if isinstance(debug_model, str) and debug_model.strip():
        return debug_model.strip()
    if not isinstance(name, str) or not name.strip():
        return "gpt-5"
    base = name.split(":", 1)[0].strip()
    for sep in ("-", "_"):
        lowered = base.lower()
        for effort in ("minimal", "low", "medium", "high", "xhigh"):
            suffix = f"{sep}{effort}"
            if lowered.endswith(suffix):
                base = base[: -len(suffix)]
                break
    mapping = {
        "gpt5": "gpt-5",
        "gpt-5-latest": "gpt-5",
        "gpt-5": "gpt-5",
        "gpt-5.1": "gpt-5.1",
        "gpt5.2": "gpt-5.2",
        "gpt-5.2": "gpt-5.2",
        "gpt-5.2-latest": "gpt-5.2",
        "gpt5.2-codex": "gpt-5.2-codex",
        "gpt-5.2-codex": "gpt-5.2-codex",
        "gpt-5.2-codex-latest": "gpt-5.2-codex",
        "gpt5-codex": "gpt-5-codex",
        "gpt-5-codex": "gpt-5-codex",
        "gpt-5-codex-latest": "gpt-5-codex",
        "gpt-5.1-codex": "gpt-5.1-codex",
        "gpt-5.1-codex-max": "gpt-5.1-codex-max",
        "codex": "codex-mini-latest",
        "codex-mini": "codex-mini-latest",
        "codex-mini-latest": "codex-mini-latest",
        "gpt-5.1-codex-mini": "gpt-5.1-codex-mini",
    }
    return mapping.get(base, base)


def _make_error_resp(msg: str, status: int):
    try:
        from flask import has_app_context
        if has_app_context():
            return None, make_response(jsonify({"error": {"message": msg}}), status)
    except:
        pass
    
    class MockResponse:
        def __init__(self, data, status_code):
            self.data = json.dumps(data).encode("utf-8")
            self.status_code = status_code
        def get_data(self, as_text=False):
            return self.data.decode("utf-8") if as_text else self.data
            
    return None, MockResponse({"error": {"message": msg}}, status)


def start_upstream_request(
    model: str,
    input_items: List[Dict[str, Any]],
    *,
    instructions: str | None = None,
    tools: List[Dict[str, Any]] | None = None,
    tool_choice: Any | None = None,
    parallel_tool_calls: bool = False,
    reasoning_param: Dict[str, Any] | None = None,
    client_session_id: str | None = None,
):
    access_token, account_id = get_effective_chatgpt_auth()
    if not access_token:
        # For CLI/Script usage, check ENV if file missing
        access_token = os.getenv("CHATGPT_ACCESS_TOKEN")
        
    if not access_token:
         return _make_error_resp("Missing Access Token. Run login or set CHATGPT_ACCESS_TOKEN.", 401)

    include: List[str] = []
    if isinstance(reasoning_param, dict):
        include.append("reasoning.encrypted_content")

    final_client_session_id = client_session_id
    if final_client_session_id is None:
        try:
            if flask_request:
                final_client_session_id = (
                    flask_request.headers.get("X-Session-Id")
                    or flask_request.headers.get("session_id")
                    or None
                )
        except Exception:
            pass
            
    session_id = ensure_session_id(instructions, input_items, final_client_session_id)

    responses_payload = {
        "model": model,
        "instructions": instructions if isinstance(instructions, str) and instructions.strip() else instructions,
        "input": input_items,
        "tools": tools or [],
        "tool_choice": tool_choice if tool_choice in ("auto", "none") or isinstance(tool_choice, dict) else "auto",
        "parallel_tool_calls": bool(parallel_tool_calls),
        "store": False,
        "stream": True,
        "prompt_cache_key": session_id,
    }
    if include:
        responses_payload["include"] = include

    if reasoning_param is not None:
        responses_payload["reasoning"] = reasoning_param

    verbose = False
    try:
        from flask import has_app_context
        if has_app_context():
            verbose = bool(current_app.config.get("VERBOSE"))
    except Exception:
        verbose = False
    
    # Caching Logic
    payload_hash = _compute_payload_hash(responses_payload)
    cached_gen = _read_from_cache_gen(payload_hash)
    if cached_gen:
        if verbose:
            _log_json(f"CACHE HIT >> Serving cached response for hash {payload_hash}", responses_payload)
        return CachedResponse(cached_gen), None

    if verbose:
        _log_json("OUTBOUND >> ChatGPT Responses API payload", responses_payload)

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "session_id": session_id,
    }

    try:
        upstream = requests.post(
            CHATGPT_RESPONSES_URL,
            headers=headers,
            json=responses_payload,
            stream=True,
            timeout=600,
        )
    except requests.RequestException as e:
        return _make_error_resp(f"Upstream ChatGPT request failed: {e}", 502)
    
    # Check if request was successful before caching
    if upstream.status_code < 400:
        # Wrap the upstream iterator to cache it
        # We use iter_content to get raw bytes
        wrapped_gen = _tee_stream(upstream.iter_content(chunk_size=None), payload_hash)
        return CachedResponse(wrapped_gen, upstream.status_code, upstream.headers), None
    else:
        # Don't cache errors
        return upstream, None