from __future__ import annotations

import functools
import inspect
import os
from typing import Any, Dict, List, Optional, Union, get_args, get_origin

from mcp.server.fastmcp import FastMCP
from .logging_utils import log_tool_call, log_resource_read


_FASTMCP_HOST = os.environ.get("FASTMCP_HOST", "127.0.0.1")
_FASTMCP_PORT = int(os.environ.get("FASTMCP_PORT", "8000"))
mcp = FastMCP("Eidosian Nexus", host=_FASTMCP_HOST, port=_FASTMCP_PORT)

_TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}
_RESOURCE_REGISTRY: Dict[str, Dict[str, Any]] = {}

_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}


def register_tool_metadata(name: str, description: str, parameters: Dict[str, Any]) -> None:
    _TOOL_REGISTRY[name] = {
        "name": name,
        "description": description,
        "parameters": parameters,
    }


def list_tool_metadata() -> List[Dict[str, Any]]:
    return list(_TOOL_REGISTRY.values())


def register_resource_metadata(uri: str, description: str) -> None:
    _RESOURCE_REGISTRY[uri] = {"uri": uri, "description": description}


def list_resource_metadata() -> List[Dict[str, Any]]:
    return list(_RESOURCE_REGISTRY.values())


def _resolve_annotation(annotation: Any) -> Dict[str, Any]:
    if annotation is inspect._empty:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is list and args:
        return {"type": "array", "items": _resolve_annotation(args[0])}
    if origin is dict and args:
        return {"type": "object"}
    if origin is Union and args:
        if type(None) in args:
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                return _resolve_annotation(non_none[0])
        return _resolve_annotation(args[0])
    if origin is list:
        return {"type": "array"}
    if origin is dict:
        return {"type": "object"}

    if annotation is Any:
        return {"type": "any"}
    if annotation in _TYPE_MAP:
        return {"type": _TYPE_MAP[annotation]}

    return {"type": "string"}


def _infer_parameters(func) -> Dict[str, Any]:
    sig = inspect.signature(func)
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        properties[name] = _resolve_annotation(param.annotation)
        if param.default is inspect._empty:
            required.append(name)
    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


import time

def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
):
    def decorator(func):
        tool_name = name or func.__name__
        desc = (description or func.__doc__ or "").strip()
        params = parameters or _infer_parameters(func)
        register_tool_metadata(tool_name, desc, params)

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                sig = inspect.signature(func)
                try:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    log_args = bound.arguments
                except Exception:
                    log_args = {"args": args, "kwargs": kwargs}

                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    log_tool_call(tool_name, log_args, result, start_time=start)
                    return result
                except Exception as e:
                    log_tool_call(tool_name, log_args, None, error=str(e), start_time=start)
                    raise
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                sig = inspect.signature(func)
                try:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    log_args = bound.arguments
                except Exception:
                    log_args = {"args": args, "kwargs": kwargs}

                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    log_tool_call(tool_name, log_args, result, start_time=start)
                    return result
                except Exception as e:
                    log_tool_call(tool_name, log_args, None, error=str(e), start_time=start)
                    raise

        if name is None:
            mcp_decorator = mcp.tool()
        else:
            mcp_decorator = mcp.tool(tool_name)
        return mcp_decorator(wrapper)

    return decorator


def resource(uri: str, description: Optional[str] = None):
    def decorator(func):
        desc = (description or func.__doc__ or "").strip()
        register_resource_metadata(uri, desc)

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    log_resource_read(uri, result, start_time=start)
                    return result
                except Exception as e:
                    log_resource_read(uri, None, error=str(e), start_time=start)
                    raise
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    log_resource_read(uri, result, start_time=start)
                    return result
                except Exception as e:
                    log_resource_read(uri, None, error=str(e), start_time=start)
                    raise

        return mcp.resource(uri)(wrapper)

    return decorator
