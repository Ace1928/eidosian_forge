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
    if list_tool_metadata():
        return
    router_modules = [
        "eidos_mcp.routers.audit",
        "eidos_mcp.routers.auth",
        "eidos_mcp.routers.consciousness",
        "eidos_mcp.routers.diagnostics",
        "eidos_mcp.routers.erais",
        "eidos_mcp.routers.gis",
        "eidos_mcp.routers.llm",
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
        "eidos_mcp.routers.plugins",  # Plugin management router
    ]
    for module_name in router_modules:
        try:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)
        except ImportError as e:
            log_debug(f"Warning: Failed to load router {module_name}: {e}")
            log_error(f"load_router:{module_name}", str(e))
        except Exception as e:
            log_debug(f"Error: Unexpected error loading router {module_name}: {e}")
            log_error(f"load_router:{module_name}", str(e))


def _sync_agent_tools() -> None:
    """Register all MCP tools into AgentForge so the agent can use them."""
    if not agent:
        log_debug("Warning: AgentForge instance is None, cannot sync tools.")
        return
    
    tools = list_tool_metadata()
    log_debug(f"Found {len(tools)} tools in registry.")
    
    count = 0
    for t in tools:
        if t.get("func"):
            agent.register_tool(t["name"], t["func"], t["description"])
            count += 1
        else:
            log_debug(f"Tool {t['name']} has no func!")
    
    if count > 0:
        log_debug(f"Synced {count} tools to AgentForge.")


def _load_plugins() -> None:
    """Load all plugins from plugin directories."""
    try:
        loaded = init_plugins(mcp)
        plugin_count = len(loaded)
        tool_count = len(list_tools())
        log_debug(f"Loaded {plugin_count} plugins with {tool_count} tools")
    except Exception as e:
        log_debug(f"Warning: Plugin loading failed: {e}")
        log_error("load_plugins", str(e))


_ensure_router_tools()
_sync_agent_tools()
_load_plugins()


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
    sources = [
        ROOT_DIR / "EIDOS_IDENTITY.md",
        ROOT_DIR / "GEMINI.md",
        FORGE_DIR / "GEMINI.md",
    ]
    body = _read_first(sources).strip()
    if _PERSONA_HEADER not in body:
        body = f"{_PERSONA_HEADER}\n\n{body}"
    return body


def _is_truthy(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _merge_accept_header(current: str, required_types: Sequence[str]) -> str:
    parts = [p.strip() for p in current.split(",") if p.strip()]
    lowered = [p.lower() for p in parts]
    for media_type in required_types:
        media_lower = media_type.lower()
        if not any(existing.startswith(media_lower) for existing in lowered):
            parts.append(media_type)
            lowered.append(media_lower)
    return ", ".join(parts)


def _get_header_value(headers: list[tuple[bytes, bytes]], name: bytes) -> Optional[str]:
    target = name.lower()
    for key, value in reversed(headers):
        if key.lower() == target:
            return value.decode("latin-1")
    return None


def _upsert_header(headers: list[tuple[bytes, bytes]], name: bytes, value: str) -> list[tuple[bytes, bytes]]:
    target = name.lower()
    updated: list[tuple[bytes, bytes]] = []
    replaced = False
    encoded_value = value.encode("latin-1")
    for key, existing_value in headers:
        if key.lower() == target:
            if not replaced:
                updated.append((target, encoded_value))
                replaced = True
            continue
        updated.append((key, existing_value))
    if not replaced:
        updated.append((target, encoded_value))
    return updated


def _headers_to_dict(headers: list[tuple[bytes, bytes]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in headers:
        result[key.decode("latin-1").lower()] = value.decode("latin-1")
    return result


def _strip_content_headers(
    headers: list[tuple[bytes, bytes]],
) -> list[tuple[bytes, bytes]]:
    stripped: list[tuple[bytes, bytes]] = []
    for key, value in headers:
        lowered = key.lower()
        if lowered in {b"content-type", b"content-length"}:
            continue
        stripped.append((key, value))
    return stripped


def _extract_jsonrpc_id(raw_body: bytes) -> Any:
    if not raw_body:
        return None
    try:
        parsed = json.loads(raw_body.decode("utf-8"))
    except Exception:
        return None
    if isinstance(parsed, dict):
        return parsed.get("id")
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and "id" in item:
                return item.get("id")
    return None


class MCPHeaderCompatibilityMiddleware:
    """Normalize missing/partial MCP transport headers for fragile clients."""

    def __init__(self, app: ASGIApp, mount_path: str) -> None:
        self.app = app
        self.mount_path = mount_path.rstrip("/") or "/"

    def _is_transport_path(self, path: str) -> bool:
        return path == self.mount_path or path == f"{self.mount_path}/"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") == "http":
            method = (scope.get("method") or "").upper()
            path = scope.get("path") or ""
            if method in {"POST", "GET"} and self._is_transport_path(path):
                headers = list(scope.get("headers", []))
                if method == "POST":
                    content_type = (_get_header_value(headers, b"content-type") or "").lower()
                    if not content_type.startswith(_JSON_MEDIA_TYPE):
                        headers = _upsert_header(headers, b"content-type", _JSON_MEDIA_TYPE)
                    accept = _get_header_value(headers, b"accept") or ""
                    merged_accept = _merge_accept_header(accept, (_JSON_MEDIA_TYPE, _SSE_MEDIA_TYPE))
                    headers = _upsert_header(headers, b"accept", merged_accept)
                elif method == "GET":
                    accept = _get_header_value(headers, b"accept") or ""
                    merged_accept = _merge_accept_header(accept, (_SSE_MEDIA_TYPE,))
                    headers = _upsert_header(headers, b"accept", merged_accept)

                scope = dict(scope)
                scope["headers"] = headers

        await self.app(scope, receive, send)


class MCPErrorResponseCompatibilityMiddleware:
    """Normalize transport errors to explicit JSON responses for strict clients."""

    def __init__(self, app: ASGIApp, mount_path: str) -> None:
        self.app = app
        self.mount_path = mount_path.rstrip("/") or "/"

    def _is_transport_path(self, path: str) -> bool:
        return path == self.mount_path or path == f"{self.mount_path}/"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = (scope.get("method") or "").upper()
        path = scope.get("path") or ""
        if method not in {"POST", "GET", "DELETE"} or not self._is_transport_path(path):
            await self.app(scope, receive, send)
            return

        response_start: dict[str, Any] | None = None
        rewrite_body = bytearray()
        should_rewrite = False
        request_body = bytearray()
        buffered_messages: list[dict[str, Any]] = []

        if method == "POST":
            while True:
                message = await receive()
                buffered_messages.append(message)
                if message.get("type") != "http.request":
                    break
                request_body.extend(message.get("body", b""))
                if not message.get("more_body", False):
                    break

        async def receive_wrapper():
            if buffered_messages:
                return buffered_messages.pop(0)
            return await receive()

        async def send_wrapper(message):
            nonlocal response_start, should_rewrite, rewrite_body
            if message["type"] == "http.response.start":
                status = int(message.get("status", 0))
                headers = list(message.get("headers", []))
                response_headers = _headers_to_dict(headers)
                response_content_type = response_headers.get("content-type", "").lower().strip()
                is_supported_type = response_content_type.startswith(_JSON_MEDIA_TYPE) or response_content_type.startswith(
                    _SSE_MEDIA_TYPE
                )
                should_rewrite = status >= 400 and not is_supported_type
                if should_rewrite:
                    response_start = message
                    return
                await send(message)
                return

            if message["type"] == "http.response.body" and should_rewrite:
                rewrite_body.extend(message.get("body", b""))
                if message.get("more_body", False):
                    return

                status_code = int((response_start or {}).get("status", 500))
                error_message = rewrite_body.decode("utf-8", errors="replace").strip()
                if not error_message:
                    error_message = f"Transport error (HTTP {status_code})"

                request_id = _extract_jsonrpc_id(bytes(request_body))
                payload = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": _JSON_ERROR_CODE,
                        "message": error_message,
                        "data": {"http_status": status_code},
                    },
                }
                payload_bytes = json.dumps(payload, ensure_ascii=True).encode("utf-8")

                start_headers = _strip_content_headers(list((response_start or {}).get("headers", [])))
                start_headers.append((b"content-type", b"application/json; charset=utf-8"))
                start_headers.append((b"content-length", str(len(payload_bytes)).encode("ascii")))

                await send(
                    {
                        "type": "http.response.start",
                        "status": status_code,
                        "headers": start_headers,
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": payload_bytes,
                        "more_body": False,
                    }
                )
                return

            await send(message)

        await self.app(scope, receive_wrapper, send_wrapper)


class MCPSessionRecoveryMiddleware:
    """Recover stale MCP sessions by clearing unknown session ids."""

    def __init__(self, app: ASGIApp, mount_path: str) -> None:
        self.app = app
        self.mount_path = mount_path.rstrip("/") or "/"

    def _is_transport_path(self, path: str) -> bool:
        return path == self.mount_path or path == f"{self.mount_path}/"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = (scope.get("method") or "").upper()
        path = scope.get("path") or ""
        if method != "POST" or not self._is_transport_path(path):
            await self.app(scope, receive, send)
            return

        headers = list(scope.get("headers", []))
        request_session_id = _get_header_value(headers, b"mcp-session-id")
        if request_session_id:
            manager = getattr(mcp, "_session_manager", None)
            known_sessions = getattr(manager, "_server_instances", None)
            is_stateless = bool(getattr(manager, "stateless", False))
            if isinstance(known_sessions, dict) and not is_stateless and request_session_id not in known_sessions:
                headers = [(k, v) for k, v in headers if k.lower() != b"mcp-session-id"]
                scope = dict(scope)
                scope["headers"] = headers
                log_debug("Recovered stale MCP session id by forcing fresh session creation")

        await self.app(scope, receive, send)


class MCPInvalidSessionCompatibilityMiddleware:
    """Reject stale MCP session ids with deterministic 400 compatibility responses."""

    def __init__(
        self, app: ASGIApp, mount_path: str, *, emit_json_error: bool
    ) -> None:
        self.app = app
        self.mount_path = mount_path.rstrip("/") or "/"
        self.emit_json_error = bool(emit_json_error)

    def _is_transport_path(self, path: str) -> bool:
        return path == self.mount_path or path == f"{self.mount_path}/"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = (scope.get("method") or "").upper()
        path = scope.get("path") or ""
        if method != "POST" or not self._is_transport_path(path):
            await self.app(scope, receive, send)
            return

        headers = list(scope.get("headers", []))
        request_session_id = _get_header_value(headers, b"mcp-session-id")
        if not request_session_id:
            await self.app(scope, receive, send)
            return

        manager = getattr(mcp, "_session_manager", None)
        known_sessions = getattr(manager, "_server_instances", None)
        is_stateless = bool(getattr(manager, "stateless", False))
        has_known_session = isinstance(known_sessions, dict) and request_session_id in known_sessions
        if is_stateless or has_known_session:
            await self.app(scope, receive, send)
            return

        buffered_messages: list[dict[str, Any]] = []
        request_body = bytearray()
        while True:
            message = await receive()
            buffered_messages.append(message)
            if message.get("type") != "http.request":
                break
            request_body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break

        request_id = _extract_jsonrpc_id(bytes(request_body))
        status_code = 400
        if self.emit_json_error:
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": _JSON_ERROR_CODE,
                    "message": "Bad Request: No valid session ID provided",
                    "data": {"http_status": status_code},
                },
            }
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            headers_out = [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
            ]
            await send({"type": "http.response.start", "status": status_code, "headers": headers_out})
            await send({"type": "http.response.body", "body": body, "more_body": False})
            log_debug("Rejected stale MCP session id with JSON compatibility error")
            return

        await send({"type": "http.response.start", "status": status_code, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        log_debug("Rejected stale MCP session id with legacy empty 400 response")


class MCPTransportAuditMiddleware:
    """Emit transport-level audit logs for MCP HTTP traffic."""

    def __init__(self, app: ASGIApp, mount_path: str) -> None:
        self.app = app
        self.mount_path = mount_path.rstrip("/") or "/"

    def _is_transport_path(self, path: str) -> bool:
        return path == self.mount_path or path == f"{self.mount_path}/"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = (scope.get("method") or "").upper()
        path = scope.get("path") or ""
        if method not in {"POST", "GET", "DELETE"} or not self._is_transport_path(path):
            await self.app(scope, receive, send)
            return

        request_headers = _headers_to_dict(list(scope.get("headers", [])))
        start = time.perf_counter()
        response_status = 0
        response_content_type = ""
        response_preview = bytearray()

        async def send_wrapper(message):
            nonlocal response_status, response_content_type, response_preview
            if message["type"] == "http.response.start":
                response_status = message.get("status", 0)
                headers = _headers_to_dict(list(message.get("headers", [])))
                response_content_type = headers.get("content-type", "")
            elif message["type"] == "http.response.body" and response_status >= 400:
                chunk = message.get("body", b"")
                remaining = max(0, 256 - len(response_preview))
                if remaining:
                    response_preview.extend(chunk[:remaining])
            await send(message)

        await self.app(scope, receive, send_wrapper)

        duration_ms = int((time.perf_counter() - start) * 1000)
        log_payload = {
            "method": method,
            "path": path,
            "status": response_status,
            "duration_ms": duration_ms,
            "request_content_type": request_headers.get("content-type", ""),
            "request_accept": request_headers.get("accept", ""),
            "request_has_session_id": bool(request_headers.get("mcp-session-id")),
            "request_has_protocol_version": bool(request_headers.get("mcp-protocol-version")),
            "response_content_type": response_content_type,
        }
        if response_status >= 400 and response_preview:
            log_payload["error_preview"] = response_preview.decode("utf-8", errors="replace")
            log_error("mcp_transport_http", json.dumps(log_payload, ensure_ascii=True))
        else:
            log_debug(f"mcp_transport_http {json.dumps(log_payload, ensure_ascii=True)}")


def _build_streamable_http_app(
    mount_path: str,
    enable_compat_headers: Optional[bool] = None,
    enable_session_recovery: Optional[bool] = None,
    enable_error_response_compat: Optional[bool] = None,
) -> ASGIApp:
    if mcp.settings.streamable_http_path != mount_path:
        mcp.settings.streamable_http_path = mount_path
        mcp._session_manager = None  # type: ignore[attr-defined]

    app = mcp.streamable_http_app()

    if enable_compat_headers is None:
        enable_compat_headers = _is_truthy(
            os.environ.get("EIDOS_MCP_ENABLE_COMPAT_HEADERS"),
            default=True,
        )
    if enable_error_response_compat is None:
        enable_error_response_compat = _is_truthy(
            os.environ.get("EIDOS_MCP_ENABLE_ERROR_RESPONSE_COMPAT"),
            default=True,
        )
    if enable_session_recovery is None:
        enable_session_recovery = _is_truthy(
            os.environ.get("EIDOS_MCP_ENABLE_SESSION_RECOVERY"),
            default=True,
        )
    enable_transport_audit = _is_truthy(
        os.environ.get("EIDOS_MCP_AUDIT_TRANSPORT"),
        default=True,
    )

    if enable_compat_headers:
        app = MCPHeaderCompatibilityMiddleware(app, mount_path)
    if enable_session_recovery:
        app = MCPSessionRecoveryMiddleware(app, mount_path)
    else:
        app = MCPInvalidSessionCompatibilityMiddleware(
            app,
            mount_path,
            emit_json_error=bool(enable_error_response_compat),
        )
    if enable_error_response_compat:
        app = MCPErrorResponseCompatibilityMiddleware(app, mount_path)
    if enable_transport_audit:
        app = MCPTransportAuditMiddleware(app, mount_path)

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
        uvicorn.run("eidos_mcp.eidos_mcp_server:app", host=host, port=port, log_level=log_level, reload=True)
    else:
        global app
        if 'app' not in globals():
            app = _build_streamable_http_app(mount_path)
        uvicorn.run(app, host=host, port=port, log_level=log_level)


@eidosian()
@resource("eidos://config", description="Global configuration snapshot (GIS).")
def resource_config() -> str:
    """Global configuration snapshot (GIS)."""
    if not gis:
        return json.dumps({})
    return json.dumps(gis.flatten(), indent=2)


@eidosian()
@resource("eidos://persona", description="Eidosian persona context.")
def resource_persona() -> str:
    """Eidosian persona context."""
    return _persona_payload()


@eidosian()
@resource("eidos://roadmap", description="Eidosian master roadmap.")
def resource_roadmap() -> str:
    """Eidosian master roadmap."""
    sources = [
        ROOT_DIR / "EIDOSIAN_MASTER_PLAN.md",
        FORGE_DIR / "REVIEW_MASTER_PLAN.md",
        FORGE_DIR / "TODO.md",
    ]
    return _read_first(sources)


@eidosian()
@resource("eidos://todo", description="Eidosian TODO list.")
def resource_todo() -> str:
    """Eidosian TODO list."""
    sources = [
        ROOT_DIR / "TODO.md",
        FORGE_DIR / "TODO.md",
    ]
    return _read_first(sources)


@eidosian()
@mcp.custom_route("/health", methods=["GET"])
async def health_check(_request):
    return JSONResponse(
        {
            "status": "ok",
            "transport": os.environ.get("EIDOS_MCP_TRANSPORT", "streamable-http"),
            "streamable_http_path": mcp.settings.streamable_http_path,
            "stateless_http": mcp.settings.stateless_http,
            "tool_count": len(list_tool_metadata()),
            "plugin_count": len(list_plugins()),
            "plugin_tool_count": len(list_tools()),
        }
    )


def _extract_bearer(header_value: Optional[str]) -> Optional[str]:
    if not header_value:
        return None
    parts = header_value.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def _verify_static_token(token: str) -> Optional[Dict[str, Any]]:
    expected = os.environ.get("EIDOS_OAUTH2_STATIC_BEARER")
    if expected and token == expected:
        return {"sub": "static", "provider": "static", "scope": "full"}
    return None


def _verify_google_token(token: str) -> Optional[Dict[str, Any]]:
    provider = os.environ.get("EIDOS_OAUTH2_PROVIDER", "").lower()
    audience = os.environ.get("EIDOS_OAUTH2_AUDIENCE")
    if provider != "google" or not audience:
        return None
    if not (id_token and google_requests):
        raise RuntimeError("google-auth not installed; cannot verify Google OAuth2 token")
    claims = id_token.verify_oauth2_token(token, google_requests.Request(), audience=audience)
    return {**claims, "provider": "google"}


@eidosian()
@mcp.custom_route("/auth/verify", methods=["GET"])
async def auth_verify(request: Request):
    token = _extract_bearer(request.headers.get("authorization")) or request.query_params.get("token")
    if not token:
        return JSONResponse({"status": "error", "error": "missing_token"}, status_code=401)
    try:
        claims = _verify_static_token(token)
        if claims is None:
            claims = _verify_google_token(token)
        if claims is None:
            return JSONResponse({"status": "error", "error": "unauthorized"}, status_code=401)
        return JSONResponse({"status": "ok", "claims": claims})
    except Exception as exc:  # pragma: no cover - defensive
        log_error("auth_verify", str(exc))
        return JSONResponse({"status": "error", "error": str(exc)}, status_code=401)


@eidosian()
def main() -> None:
    """Run the Eidosian MCP server (streamable-http)."""
    transport = os.environ.get("EIDOS_MCP_TRANSPORT", "streamable-http")
    mount_path = os.environ.get("EIDOS_MCP_MOUNT_PATH", "/mcp")
    log_debug(f"Starting Eidosian MCP Server (Transport: {transport}, Mount: {mount_path})...")
    log_startup(transport)
    try:
        normalized_transport = transport.strip().lower().replace("_", "-")
        if normalized_transport == "streamable-http":
            _run_streamable_http_server(mount_path)
        else:
            mcp.run(transport=transport, mount_path=mount_path)
    except Exception as e:
        msg = f"Critical Error: MCP Server failed to start: {e}"
        log_debug(msg)
        log_error("startup", msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
