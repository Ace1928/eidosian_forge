from __future__ import annotations

import importlib
import sys
from typing import Any, List, Optional

from .core import list_tool_metadata, mcp
from .logging_utils import log_debug, log_error
from .plugins import init_plugins, list_plugins, list_tools
from . import state as _forge_state

class ForgeManager:
    """
    Centralized manager for Eidosian Forge components within the MCP server.
    Handles registration of routers, synchronization of tools, and plugin lifecycles.
    """

    def __init__(self, mcp_instance: Any = None):
        self.mcp = mcp_instance or mcp
        self.agent = _forge_state.agent
        self.router_modules: List[str] = [
            "eidos_mcp.routers.article",
            "eidos_mcp.routers.audit",
            "eidos_mcp.routers.auth",
            "eidos_mcp.routers.autonomy",
            "eidos_mcp.routers.code",
            "eidos_mcp.routers.consciousness",
            "eidos_mcp.routers.diagnostics",
            "eidos_mcp.routers.erais",
            "eidos_mcp.routers.figlet",
            "eidos_mcp.routers.game",
            "eidos_mcp.routers.gis",
            "eidos_mcp.routers.glyph",
            "eidos_mcp.routers.llm",
            "eidos_mcp.routers.knowledge",
            "eidos_mcp.routers.learner",
            "eidos_mcp.routers.memory",
            "eidos_mcp.routers.moltbook",
            "eidos_mcp.routers.narrative",
            "eidos_mcp.routers.nexus",
            "eidos_mcp.routers.prompt",
            "eidos_mcp.routers.refactor",
            "eidos_mcp.routers.repo",
            "eidos_mcp.routers.sms",
            "eidos_mcp.routers.system",
            "eidos_mcp.routers.terminal",
            "eidos_mcp.routers.tika",
            "eidos_mcp.routers.tiered_memory",
            "eidos_mcp.routers.types",
            "eidos_mcp.routers.web_interface",
            "eidos_mcp.routers.word_forge",
            "eidos_mcp.routers.plugins",
        ]

    def register_routers(self, force_reload: bool = False) -> None:
        """Dynamically import and register all configured MCP routers."""
        for module_name in self.router_modules:
            try:
                if force_reload and module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
                else:
                    importlib.import_module(module_name)
            except ImportError as e:
                log_debug(f"Warning: Failed to load router {module_name}: {e}")
                log_error(f"load_router:{module_name}", str(e))
            except Exception as e:
                log_debug(f"Error: Unexpected error loading router {module_name}: {e}")
                log_error(f"load_router:{module_name}", str(e))

    def sync_agent_tools(self) -> None:
        """Register all registered MCP tools into AgentForge for agent use."""
        if not self.agent:
            log_debug("Warning: AgentForge instance is None, cannot sync tools.")
            return

        tools = list_tool_metadata()
        log_debug(f"Found {len(tools)} tools in registry.")

        count = 0
        for t in tools:
            if t.get("func"):
                self.agent.register_tool(t["name"], t["func"], t["description"])
                count += 1
            else:
                log_debug(f"Tool {t['name']} has no func!")

        if count > 0:
            log_debug(f"Synced {count} tools to AgentForge.")

    def load_plugins(self) -> None:
        """Initialize and load all discovered plugins."""
        try:
            loaded = init_plugins(self.mcp)
            plugin_count = len(loaded)
            tool_count = len(list_tools())
            log_debug(f"Loaded {plugin_count} plugins with {tool_count} tools")
        except Exception as e:
            log_debug(f"Warning: Plugin loading failed: {e}")
            log_error("load_plugins", str(e))

    def initialize_all(self, force_reload: bool = False) -> None:
        """Perform full initialization sequence."""
        self.register_routers(force_reload=force_reload)
        self.sync_agent_tools()
        self.load_plugins()

# Default singleton instance
manager = ForgeManager()
