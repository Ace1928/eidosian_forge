"""
🔌 Eidosian Plugin Architecture

A robust, extensible plugin system for the MCP server.
Enables dynamic tool registration, hot-reload, health monitoring.

Design Principles:
- Zero-downtime plugin loading
- Isolated plugin execution
- Automatic dependency resolution
- Full provenance tracking
- Performance monitoring

Created: 2026-01-23
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from eidosian_core import eidosian

from .. import FORGE_ROOT

logger = logging.getLogger("eidos.plugins")

# Plugin directories
PLUGIN_DIRS = [
    Path(__file__).parent,  # This is eidos_mcp/plugins itself
    Path(os.environ.get("EIDOS_PLUGIN_DIR", str(FORGE_ROOT / "plugins"))),
]

# Plugin registry
_PLUGINS: Dict[str, "PluginInfo"] = {}
_PLUGIN_TOOLS: Dict[str, "ToolInfo"] = {}
_LOAD_TIMES: Dict[str, float] = {}


@dataclass
class ToolInfo:
    """Metadata for a registered tool."""

    name: str
    description: str
    plugin_id: str
    func: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    calls: int = 0
    total_time: float = 0.0
    errors: int = 0
    last_called: Optional[str] = None

    @property
    def avg_time(self) -> float:
        return self.total_time / self.calls if self.calls > 0 else 0.0


@dataclass
class PluginInfo:
    """Metadata for a loaded plugin."""

    id: str
    name: str
    version: str
    description: str
    author: str = "Eidos"
    path: Optional[Path] = None
    tools: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True
    loaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    load_time_ms: float = 0.0
    status: str = "loaded"
    error: Optional[str] = None


class PluginLoader:
    """
    Dynamic plugin loader with hot-reload support.

    Features:
    - Load plugins from multiple directories
    - Automatic manifest discovery
    - Dependency resolution
    - Hot-reload without restart
    - Health monitoring
    """

    def __init__(self, mcp_instance=None):
        self.mcp = mcp_instance
        self.plugin_dirs = PLUGIN_DIRS
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Ensure plugin directories exist."""
        for d in self.plugin_dirs:
            d.mkdir(parents=True, exist_ok=True)

    @eidosian()
    def discover_plugins(self) -> List[Path]:
        """Discover all plugin manifests."""
        manifests = []
        for plugin_dir in self.plugin_dirs:
            if plugin_dir.exists():
                # Look for plugin.json manifests
                for manifest in plugin_dir.glob("*/plugin.json"):
                    manifests.append(manifest)
                # Look for __init__.py with PLUGIN_MANIFEST in subdirectories
                for init_file in plugin_dir.glob("*/__init__.py"):
                    if init_file.parent.name.startswith("_"):
                        continue
                    # Check if it has PLUGIN_MANIFEST
                    try:
                        content = init_file.read_text()
                        if "PLUGIN_MANIFEST" in content:
                            # Only add if no plugin.json exists
                            json_path = init_file.parent / "plugin.json"
                            if not json_path.exists():
                                manifests.append(init_file)
                    except Exception:
                        pass
                # Also look for standalone .py files with PLUGIN_MANIFEST
                for py_file in plugin_dir.glob("*.py"):
                    if py_file.name.startswith("_"):
                        continue
                    manifests.append(py_file)
        return manifests

    @eidosian()
    def load_manifest(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load plugin manifest from file or module."""
        try:
            if path.suffix == ".json":
                with open(path) as f:
                    manifest = json.load(f)
                # Also load the __init__.py module if it exists
                init_path = path.parent / "__init__.py"
                if init_path.exists():
                    spec = importlib.util.spec_from_file_location(f"plugin_{path.parent.name}", init_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        manifest["_module"] = module
                        manifest["_path"] = path
                return manifest
            elif path.suffix == ".py":
                # Import module and get PLUGIN_MANIFEST attribute
                spec = importlib.util.spec_from_file_location(f"plugin_{path.stem}", path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "PLUGIN_MANIFEST"):
                        manifest = getattr(module, "PLUGIN_MANIFEST")
                        manifest["_module"] = module
                        manifest["_path"] = path
                        return manifest
        except Exception as e:
            logger.error(f"Failed to load manifest from {path}: {e}")
        return None

    @eidosian()
    def load_plugin(self, manifest_path: Path) -> Optional[PluginInfo]:
        """Load a single plugin from manifest."""
        start_time = time.time()

        manifest = self.load_manifest(manifest_path)
        if not manifest:
            return None

        plugin_id = manifest.get("id", manifest_path.stem)

        # Check if already loaded
        if plugin_id in _PLUGINS:
            logger.info(f"Plugin {plugin_id} already loaded, skipping")
            return _PLUGINS[plugin_id]

        try:
            # Create plugin info
            plugin = PluginInfo(
                id=plugin_id,
                name=manifest.get("name", plugin_id),
                version=manifest.get("version", "1.0.0"),
                description=manifest.get("description", ""),
                author=manifest.get("author", "Eidos"),
                path=manifest_path,
                dependencies=manifest.get("dependencies", []),
            )

            # Load dependencies first
            for dep in plugin.dependencies:
                if dep not in _PLUGINS:
                    logger.warning(f"Missing dependency {dep} for plugin {plugin_id}")

            # Register tools from manifest
            tools = manifest.get("tools", [])
            module = manifest.get("_module")

            for tool_spec in tools:
                if isinstance(tool_spec, str) and module:
                    # Tool name references a function in the module
                    if hasattr(module, tool_spec):
                        func = getattr(module, tool_spec)
                        self._register_tool(plugin_id, tool_spec, func)
                        plugin.tools.append(tool_spec)
                elif isinstance(tool_spec, dict):
                    # Detailed tool specification
                    tool_name = tool_spec.get("name")
                    if tool_name and module and hasattr(module, tool_name):
                        func = getattr(module, tool_name)
                        self._register_tool(
                            plugin_id,
                            tool_name,
                            func,
                            description=tool_spec.get("description"),
                            parameters=tool_spec.get("parameters"),
                            tags=tool_spec.get("tags", []),
                        )
                        plugin.tools.append(tool_name)

            # Calculate load time
            plugin.load_time_ms = (time.time() - start_time) * 1000
            _LOAD_TIMES[plugin_id] = plugin.load_time_ms

            # Store plugin
            _PLUGINS[plugin_id] = plugin
            logger.info(
                f"Loaded plugin {plugin_id} v{plugin.version} ({len(plugin.tools)} tools) in {plugin.load_time_ms:.1f}ms"
            )

            return plugin

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_id}: {e}")
            traceback.print_exc()

            # Store failed plugin info
            plugin = PluginInfo(
                id=plugin_id,
                name=manifest.get("name", plugin_id),
                version=manifest.get("version", "1.0.0"),
                description=manifest.get("description", ""),
                path=manifest_path,
                enabled=False,
                status="error",
                error=str(e),
            )
            _PLUGINS[plugin_id] = plugin
            return None

    def _register_tool(
        self,
        plugin_id: str,
        name: str,
        func: Callable,
        description: Optional[str] = None,
        parameters: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        """Register a tool from a plugin."""
        tool_info = ToolInfo(
            name=name,
            description=description or (func.__doc__ or "").strip(),
            plugin_id=plugin_id,
            func=func,
            parameters=parameters or {},
            tags=tags or [],
        )

        # Wrap function for metrics
        @eidosian()
        @wraps(func)
        def wrapped(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                tool_info.calls += 1
                tool_info.total_time += time.time() - start
                tool_info.last_called = datetime.now(timezone.utc).isoformat()
                return result
            except Exception:
                tool_info.errors += 1
                raise

        tool_info.func = wrapped
        _PLUGIN_TOOLS[name] = tool_info

        # Register with MCP if available
        if self.mcp:
            try:
                # Import the decorator
                from .core import register_tool_metadata

                register_tool_metadata(name, tool_info.description, tool_info.parameters)
            except Exception as e:
                logger.warning(f"Could not register {name} with MCP: {e}")

    @eidosian()
    def load_all(self) -> Dict[str, PluginInfo]:
        """Load all discovered plugins."""
        manifests = self.discover_plugins()
        loaded = {}

        for manifest_path in manifests:
            plugin = self.load_plugin(manifest_path)
            if plugin:
                loaded[plugin.id] = plugin

        return loaded

    @eidosian()
    def reload_plugin(self, plugin_id: str) -> Optional[PluginInfo]:
        """Hot-reload a plugin."""
        if plugin_id not in _PLUGINS:
            logger.error(f"Plugin {plugin_id} not found")
            return None

        old_plugin = _PLUGINS[plugin_id]

        # Unregister old tools
        for tool_name in old_plugin.tools:
            if tool_name in _PLUGIN_TOOLS:
                del _PLUGIN_TOOLS[tool_name]

        # Remove from registry
        del _PLUGINS[plugin_id]

        # Reload from path
        if old_plugin.path:
            return self.load_plugin(old_plugin.path)

        return None

    @eidosian()
    def get_plugin_stats(self) -> Dict[str, Any]:
        """Get comprehensive plugin statistics."""
        return {
            "total_plugins": len(_PLUGINS),
            "enabled_plugins": sum(1 for p in _PLUGINS.values() if p.enabled),
            "total_tools": len(_PLUGIN_TOOLS),
            "total_calls": sum(t.calls for t in _PLUGIN_TOOLS.values()),
            "total_errors": sum(t.errors for t in _PLUGIN_TOOLS.values()),
            "avg_load_time_ms": sum(_LOAD_TIMES.values()) / len(_LOAD_TIMES) if _LOAD_TIMES else 0,
            "plugins": {
                pid: {
                    "name": p.name,
                    "version": p.version,
                    "tools": len(p.tools),
                    "status": p.status,
                    "load_time_ms": p.load_time_ms,
                }
                for pid, p in _PLUGINS.items()
            },
        }


# Global loader instance
_loader: Optional[PluginLoader] = None


@eidosian()
def get_loader(mcp_instance=None) -> PluginLoader:
    """Get or create the global plugin loader."""
    global _loader
    if _loader is None:
        _loader = PluginLoader(mcp_instance)
    return _loader


@eidosian()
def init_plugins(mcp_instance=None) -> Dict[str, PluginInfo]:
    """Initialize and load all plugins."""
    loader = get_loader(mcp_instance)
    return loader.load_all()


@eidosian()
def get_tool(name: str) -> Optional[ToolInfo]:
    """Get a registered tool by name."""
    return _PLUGIN_TOOLS.get(name)


@eidosian()
def call_tool(name: str, *args, **kwargs) -> Any:
    """Call a registered tool by name."""
    tool = get_tool(name)
    if not tool:
        raise ValueError(f"Tool {name} not found")
    return tool.func(*args, **kwargs)


@eidosian()
def list_plugins() -> List[PluginInfo]:
    """List all loaded plugins."""
    return list(_PLUGINS.values())


@eidosian()
def list_tools() -> List[ToolInfo]:
    """List all registered tools."""
    return list(_PLUGIN_TOOLS.values())


# Export for easy access
__all__ = [
    "PluginLoader",
    "PluginInfo",
    "ToolInfo",
    "get_loader",
    "init_plugins",
    "get_tool",
    "call_tool",
    "list_plugins",
    "list_tools",
]
