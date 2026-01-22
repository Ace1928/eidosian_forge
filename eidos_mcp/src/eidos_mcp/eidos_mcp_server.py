from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import importlib
import sys

from starlette.responses import JSONResponse

from .core import mcp, resource, list_tool_metadata
from .logging_utils import log_startup, log_error
from .state import gis, llm, refactor, agent, ROOT_DIR, FORGE_DIR
from . import routers as _routers  # noqa: F401


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
        "eidos_mcp.routers.types",
    ]
    for module_name in router_modules:
        try:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)
        except ImportError as e:
            print(f"Warning: Failed to load router {module_name}: {e}", file=sys.stderr)
            log_error(f"load_router:{module_name}", str(e))
        except Exception as e:
            print(f"Error: Unexpected error loading router {module_name}: {e}", file=sys.stderr)
            log_error(f"load_router:{module_name}", str(e))


_ensure_router_tools()


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


@resource("eidos://config", description="Global configuration snapshot (GIS).")
def resource_config() -> str:
    """Global configuration snapshot (GIS)."""
    if not gis:
        return json.dumps({})
    return json.dumps(gis.flatten(), indent=2)


@resource("eidos://persona", description="Eidosian persona context.")
def resource_persona() -> str:
    """Eidosian persona context."""
    return _persona_payload()


@resource("eidos://roadmap", description="Eidosian master roadmap.")
def resource_roadmap() -> str:
    """Eidosian master roadmap."""
    sources = [
        ROOT_DIR / "EIDOSIAN_MASTER_PLAN.md",
        FORGE_DIR / "REVIEW_MASTER_PLAN.md",
        FORGE_DIR / "TODO.md",
    ]
    return _read_first(sources)


@resource("eidos://todo", description="Eidosian TODO list.")
def resource_todo() -> str:
    """Eidosian TODO list."""
    sources = [
        ROOT_DIR / "TODO.md",
        FORGE_DIR / "TODO.md",
    ]
    return _read_first(sources)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(_request):
    return JSONResponse({"status": "ok"})


def main() -> None:
    """Run the Eidosian MCP server (stdio by default)."""
    transport = os.environ.get("EIDOS_MCP_TRANSPORT", "stdio")
    print(f"Starting Eidosian MCP Server (Transport: {transport})...", file=sys.stderr)
    log_startup(transport)
    try:
        mcp.run(transport=transport)
    except Exception as e:
        msg = f"Critical Error: MCP Server failed to start: {e}"
        print(msg, file=sys.stderr)
        log_error("startup", msg)
        sys.exit(1)
