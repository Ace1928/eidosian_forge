from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Dict, Any, Optional

import importlib
import sys

from .logging_utils import setup_logging, log_startup, log_error, log_debug
setup_logging()

from starlette.responses import JSONResponse
from starlette.requests import Request

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
        "eidos_mcp.routers.diagnostics",
        "eidos_mcp.routers.gis",
        "eidos_mcp.routers.knowledge",
        "eidos_mcp.routers.memory",
        "eidos_mcp.routers.nexus",
        "eidos_mcp.routers.refactor",
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
    return JSONResponse({"status": "ok"})


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
        mcp.run(transport=transport, mount_path=mount_path)
    except Exception as e:
        msg = f"Critical Error: MCP Server failed to start: {e}"
        log_debug(msg)
        log_error("startup", msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
