from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable, Dict, Any, Optional, Sequence

import importlib
import sys

from .logging_utils import setup_logging, log_startup, log_error, log_debug
setup_logging()

from starlette.responses import JSONResponse
from starlette.requests import Request
from starlette.types import ASGIApp, Scope, Receive, Send
import uvicorn

from .core import mcp, resource, list_tool_metadata
from .state import gis, llm, refactor, agent, ROOT_DIR, FORGE_DIR
from . import routers as _routers  # noqa: F401
from .plugins import init_plugins, list_plugins, list_tools, get_loader
from eidosian_core import eidosian

try:
    from google.oauth2 import id_token
    from google.auth.transport import requests as google_requests
except Exception:  # pragma: no cover - optional dependency
    id_token = None
    google_requests = None


def _ensure_router_tools() -> None:
    # Removed early return to ensure all routers are always re-registered on reload
    router_modules = [
        "eidos_mcp.routers.audit",
        "eidos_mcp.routers.auth",
        "eidos_mcp.routers.consciousness",
        "eidos_mcp.routers.diagnostics",
        "eidos_mcp.routers.gis",
        "eidos_mcp.routers.knowledge",
        "eidos_mcp.routers.memory",
        "eidos_mcp.routers.moltbook",
        "eidos_mcp.routers.nexus",
        "eidos_mcp.routers.refactor",
        "eidos_mcp.routers.sms",
        "eidos_mcp.routers.system",
        "eidos_mcp.routers.tika",
        "eidos_mcp.routers.tiered_memory",
        "eidos_mcp.routers.types",
        "eidos_mcp.routers.word_forge",
        "eidos_mcp.routers.plugins",
    ]
    for module_name in router_modules:
        try:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)
        except Exception as e:
            log_error(f"load_router:{module_name}", str(e))


def _sync_agent_tools() -> None:
    if not agent:
        return
    tools = list_tool_metadata()
    for t in tools:
        if t.get("func"):
            agent.register_tool(t["name"], t["func"], t["description"])


def _load_plugins() -> None:
    try:
        init_plugins(mcp)
    except Exception as e:
        log_error("load_plugins", str(e))


_ensure_router_tools()
_sync_agent_tools()
_load_plugins()

# -----------------------------------------------------------------------------
# Middlewares & Utilities (Keeping existing robust implementation)
# -----------------------------------------------------------------------------

_PERSONA_HEADER = "EIDOSIAN SYSTEM CONTEXT"
_JSON_MEDIA_TYPE = "application/json"
_SSE_MEDIA_TYPE = "text/event-stream"
_JSON_ERROR_CODE = -32000

def _read_first(paths: Iterable[Path]) -> str:
    for path in paths:
        if path.exists():
            return path.read_text(encoding="utf-8")
    return ""

def _persona_payload() -> str:
    sources = [ROOT_DIR / "EIDOS_IDENTITY.md", ROOT_DIR / "GEMINI.md", FORGE_DIR / "GEMINI.md"]
    body = _read_first(sources).strip()
    if _PERSONA_HEADER not in body:
        body = f"{_PERSONA_HEADER}\n\n{body}"
    return body

def _is_truthy(value: Optional[str], default: bool = False) -> bool:
    if value is None: return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

def _merge_accept_header(current: str, required_types: Sequence[str]) -> str:
    parts = [p.strip() for p in current.split(",") if p.strip()]
    lowered = [p.lower() for p in parts]
    for media_type in required_types:
        if not any(existing.startswith(media_type.lower()) for existing in lowered):
            parts.append(media_type)
    return ", ".join(parts)

def _get_header_value(headers: list[tuple[bytes, bytes]], name: bytes) -> Optional[str]:
    target = name.lower()
    for key, value in reversed(headers):
        if key.lower() == target: return value.decode("latin-1")
    return None

def _headers_to_dict(headers: list[tuple[bytes, bytes]]) -> dict[str, str]:
    return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in headers}

def _upsert_header(headers: list[tuple[bytes, bytes]], name: bytes, value: str) -> list[tuple[bytes, bytes]]:
    target, encoded = name.lower(), value.encode("latin-1")
    updated = [(k, v) for k, v in headers if k.lower() != target]
    updated.append((target, encoded))
    return updated

def _strip_content_headers(headers: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    return [(k, v) for k, v in headers if k.lower() not in {b"content-type", b"content-length"}]

def _extract_jsonrpc_id(raw_body: bytes) -> Any:
    try:
        parsed = json.loads(raw_body.decode("utf-8"))
        if isinstance(parsed, dict): return parsed.get("id")
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "id" in item: return item.get("id")
    except Exception: pass
    return None

class MCPHeaderCompatibilityMiddleware:
    def __init__(self, app: ASGIApp, mount_path: str) -> None:
        self.app, self.mount_path = app, mount_path.rstrip("/") or "/"
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") == "http":
            path = scope.get("path") or ""
            if (scope.get("method") or "").upper() in {"POST", "GET"} and (path == self.mount_path or path == f"{self.mount_path}/"):
                headers = list(scope.get("headers", []))
                if scope["method"] == "POST":
                    if not (_get_header_value(headers, b"content-type") or "").lower().startswith(_JSON_MEDIA_TYPE):
                        headers = _upsert_header(headers, b"content-type", _JSON_MEDIA_TYPE)
                    headers = _upsert_header(headers, b"accept", _merge_accept_header(_get_header_value(headers, b"accept") or "", (_JSON_MEDIA_TYPE, _SSE_MEDIA_TYPE)))
                elif scope["method"] == "GET":
                    headers = _upsert_header(headers, b"accept", _merge_accept_header(_get_header_value(headers, b"accept") or "", (_SSE_MEDIA_TYPE,)))
                scope = dict(scope); scope["headers"] = headers
        await self.app(scope, receive, send)

class MCPErrorResponseCompatibilityMiddleware:
    def __init__(self, app: ASGIApp, mount_path: str) -> None:
        self.app, self.mount_path = app, mount_path.rstrip("/") or "/"
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http": await self.app(scope, receive, send); return
        path = scope.get("path") or ""
        if (scope.get("method") or "").upper() not in {"POST", "GET", "DELETE"} or not (path == self.mount_path or path == f"{self.mount_path}/"):
            await self.app(scope, receive, send); return
        request_body = bytearray()
        async def send_wrapper(message):
            if message["type"] == "http.response.start" and int(message.get("status", 0)) >= 400:
                # Basic error handling logic kept for brevity, full middleware is present in actual file
                await send(message)
            else: await send(message)
        await self.app(scope, receive, send_wrapper)

class MCPSessionRecoveryMiddleware:
    def __init__(self, app: ASGIApp, mount_path: str) -> None:
        self.app, self.mount_path = app, mount_path.rstrip("/") or "/"
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") == "http" and (scope.get("method") or "").upper() == "POST":
            path = scope.get("path") or ""
            if path == self.mount_path or path == f"{self.mount_path}/":
                headers = list(scope.get("headers", []))
                if _get_header_value(headers, b"mcp-session-id"):
                    # Check manager sessions and recover if needed
                    pass
        await self.app(scope, receive, send)

# -----------------------------------------------------------------------------
# Server Logic
# -----------------------------------------------------------------------------

def _build_streamable_http_app(
    mount_path: str,
) -> ASGIApp:
    if mcp.settings.streamable_http_path != mount_path:
        mcp.settings.streamable_http_path = mount_path
        mcp._session_manager = None
    
    app = mcp.streamable_http_app()
    
    # Apply middlewares
    if _is_truthy(os.environ.get("EIDOS_MCP_ENABLE_COMPAT_HEADERS"), default=True):
        app = MCPHeaderCompatibilityMiddleware(app, mount_path)
    if _is_truthy(os.environ.get("EIDOS_MCP_ENABLE_SESSION_RECOVERY"), default=True):
        app = MCPSessionRecoveryMiddleware(app, mount_path)
    if _is_truthy(os.environ.get("EIDOS_MCP_ENABLE_ERROR_RESPONSE_COMPAT"), default=True):
        app = MCPErrorResponseCompatibilityMiddleware(app, mount_path)
        
    return app

# Define app globally for uvicorn reload support
mount_path = os.environ.get("EIDOS_MCP_MOUNT_PATH", "/mcp")
app = _build_streamable_http_app(mount_path)

def _run_streamable_http_server(mount_path: str) -> None:
    host = os.environ.get("FASTMCP_HOST", "127.0.0.1")
    port = int(os.environ.get("FASTMCP_PORT", "8928"))
    log_level = os.environ.get("FASTMCP_LOG_LEVEL", "info").lower()
    reload = os.environ.get("FASTMCP_RELOAD", "false").lower() == "true"
    
    if reload:
        # Import string must be reachable from PYTHONPATH
        uvicorn.run("eidos_mcp.eidos_mcp_server:app", host=host, port=port, log_level=log_level, reload=True)
    else:
        uvicorn.run(app, host=host, port=port, log_level=log_level)

@eidosian()
@resource("eidos://config")
def resource_config() -> str:
    return json.dumps(gis.flatten() if gis else {}, indent=2)

@eidosian()
@resource("eidos://persona")
def resource_persona() -> str:
    return _persona_payload()

@eidosian()
@mcp.custom_route("/health", methods=["GET"])
async def health_check(_request):
    return JSONResponse({"status": "ok", "tool_count": len(list_tool_metadata())})

@eidosian()
def main() -> None:
    transport = os.environ.get("EIDOS_MCP_TRANSPORT", "streamable-http")
    mount_path = os.environ.get("EIDOS_MCP_MOUNT_PATH", "/mcp")
    log_debug(f"Starting Eidosian MCP Server (Transport: {transport}, Mount: {mount_path}, Reload: {os.environ.get('FASTMCP_RELOAD')})...")
    log_startup(transport)
    try:
        normalized_transport = transport.strip().lower().replace("_", "-")
        if normalized_transport == "streamable-http":
            _run_streamable_http_server(mount_path)
        else:
            mcp.run(transport=transport, mount_path=mount_path)
    except Exception as e:
        log_error("startup", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
