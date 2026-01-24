from eidosian_core import eidosian
"""
Plugin Core Module - Tool Registration Support.

This module provides the registration mechanism for plugin tools
to integrate with the main MCP server.
"""

from typing import Any, Callable, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Registry for tool metadata
_TOOL_METADATA: Dict[str, Dict[str, Any]] = {}


@eidosian()
def register_tool_metadata(
    name: str,
    description: str,
    parameters: Optional[Dict[str, Any]] = None,
    returns: Optional[str] = None,
    examples: Optional[List[str]] = None,
) -> None:
    """
    Register metadata for a plugin tool.
    
    This metadata is used by the MCP server to expose the tool
    to connected clients with proper documentation.
    
    Args:
        name: Tool name (must be unique across all plugins)
        description: Human-readable description of what the tool does
        parameters: JSON Schema for tool parameters
        returns: Description of return value
        examples: List of usage examples
    """
    if name in _TOOL_METADATA:
        logger.warning(f"Tool {name} already registered, overwriting")
    
    _TOOL_METADATA[name] = {
        "name": name,
        "description": description,
        "parameters": parameters or {},
        "returns": returns,
        "examples": examples or [],
    }
    logger.debug(f"Registered tool metadata: {name}")


@eidosian()
def get_tool_metadata(name: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a registered tool."""
    return _TOOL_METADATA.get(name)


@eidosian()
def list_registered_tools() -> List[str]:
    """List all registered tool names."""
    return list(_TOOL_METADATA.keys())


@eidosian()
def get_all_tool_metadata() -> Dict[str, Dict[str, Any]]:
    """Get metadata for all registered tools."""
    return _TOOL_METADATA.copy()


@eidosian()
def unregister_tool(name: str) -> bool:
    """
    Unregister a tool.
    
    Returns:
        True if tool was unregistered, False if not found
    """
    if name in _TOOL_METADATA:
        del _TOOL_METADATA[name]
        logger.debug(f"Unregistered tool: {name}")
        return True
    return False


@eidosian()
def clear_all_tools() -> int:
    """
    Clear all registered tools.
    
    Returns:
        Number of tools that were cleared
    """
    count = len(_TOOL_METADATA)
    _TOOL_METADATA.clear()
    return count


__all__ = [
    "register_tool_metadata",
    "get_tool_metadata", 
    "list_registered_tools",
    "get_all_tool_metadata",
    "unregister_tool",
    "clear_all_tools",
]
