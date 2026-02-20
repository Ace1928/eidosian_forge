"""
🔌 Plugin Management Router

MCP tools for managing plugins, discovering tools, and monitoring performance.

Created: 2026-01-23
"""

from __future__ import annotations

import json
from typing import Optional

from eidosian_core import eidosian

from ..core import tool
from ..plugins import call_tool, get_loader, get_tool, list_plugins, list_tools


@tool(
    description="List all loaded plugins with their status and tool counts.",
    parameters={
        "type": "object",
        "properties": {"include_tools": {"type": "boolean", "description": "Include list of tools for each plugin"}},
        "required": [],
    },
)
@eidosian()
def plugin_list(include_tools: bool = False) -> str:
    """List all loaded plugins."""
    plugins = list_plugins()

    result = []
    for p in plugins:
        info = {
            "id": p.id,
            "name": p.name,
            "version": p.version,
            "status": p.status,
            "tool_count": len(p.tools),
            "load_time_ms": round(p.load_time_ms, 2),
        }
        if include_tools:
            info["tools"] = p.tools
        result.append(info)

    return json.dumps(result, indent=2)


@tool(
    description="Get detailed statistics about the plugin system.",
    parameters={"type": "object", "properties": {}, "required": []},
)
@eidosian()
def plugin_stats() -> str:
    """Get plugin system statistics."""
    loader = get_loader()
    stats = loader.get_plugin_stats()
    return json.dumps(stats, indent=2)


@tool(
    description="List all available tools across all plugins.",
    parameters={
        "type": "object",
        "properties": {
            "filter_tag": {"type": "string", "description": "Filter tools by tag"},
            "filter_plugin": {"type": "string", "description": "Filter tools by plugin ID"},
        },
        "required": [],
    },
)
@eidosian()
def tool_list(filter_tag: Optional[str] = None, filter_plugin: Optional[str] = None) -> str:
    """List all available tools."""
    tools = list_tools()

    if filter_tag:
        tools = [t for t in tools if filter_tag in t.tags]
    if filter_plugin:
        tools = [t for t in tools if t.plugin_id == filter_plugin]

    result = []
    for t in tools:
        result.append(
            {
                "name": t.name,
                "plugin": t.plugin_id,
                "description": t.description[:100] + "..." if len(t.description) > 100 else t.description,
                "calls": t.calls,
                "avg_time_ms": round(t.avg_time * 1000, 2),
                "errors": t.errors,
                "tags": t.tags,
            }
        )

    return json.dumps(result, indent=2)


@tool(
    description="Get detailed information about a specific tool.",
    parameters={
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Tool name"}},
        "required": ["name"],
    },
)
@eidosian()
def tool_info(name: str) -> str:
    """Get detailed tool information."""
    t = get_tool(name)
    if not t:
        return json.dumps({"error": f"Tool '{name}' not found"})

    return json.dumps(
        {
            "name": t.name,
            "plugin": t.plugin_id,
            "description": t.description,
            "version": t.version,
            "parameters": t.parameters,
            "tags": t.tags,
            "stats": {
                "calls": t.calls,
                "total_time_s": round(t.total_time, 3),
                "avg_time_ms": round(t.avg_time * 1000, 2),
                "errors": t.errors,
                "last_called": t.last_called,
            },
        },
        indent=2,
    )


@tool(
    description="Reload a plugin to pick up changes.",
    parameters={
        "type": "object",
        "properties": {"plugin_id": {"type": "string", "description": "Plugin ID to reload"}},
        "required": ["plugin_id"],
    },
)
@eidosian()
def plugin_reload(plugin_id: str) -> str:
    """Reload a plugin."""
    loader = get_loader()
    result = loader.reload_plugin(plugin_id)

    if result:
        return json.dumps(
            {
                "status": "success",
                "plugin": result.id,
                "version": result.version,
                "tools": result.tools,
                "load_time_ms": round(result.load_time_ms, 2),
            }
        )
    else:
        return json.dumps({"status": "error", "message": f"Failed to reload plugin {plugin_id}"})


@tool(description="Discover and load any new plugins.", parameters={"type": "object", "properties": {}, "required": []})
@eidosian()
def plugin_discover() -> str:
    """Discover and load new plugins."""
    loader = get_loader()
    before_count = len(list_plugins())

    loaded = loader.load_all()
    after_count = len(list_plugins())
    new_count = after_count - before_count

    return json.dumps(
        {
            "status": "success",
            "new_plugins_loaded": new_count,
            "total_plugins": after_count,
            "loaded": [{"id": p.id, "name": p.name, "tools": len(p.tools)} for p in loaded.values()],
        },
        indent=2,
    )


@tool(
    description="Call a tool by name with arguments (for dynamic tool invocation).",
    parameters={
        "type": "object",
        "properties": {
            "tool_name": {"type": "string", "description": "Name of the tool to call"},
            "args": {"type": "object", "description": "Arguments to pass to the tool"},
        },
        "required": ["tool_name"],
    },
)
@eidosian()
def tool_invoke(tool_name: str, args: Optional[dict] = None) -> str:
    """Dynamically invoke a tool by name."""
    try:
        result = call_tool(tool_name, **(args or {}))
        return json.dumps({"status": "success", "tool": tool_name, "result": result})
    except Exception as e:
        return json.dumps({"status": "error", "tool": tool_name, "error": str(e)})
