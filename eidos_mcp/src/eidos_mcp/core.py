from __future__ import annotations

import functools
import inspect
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Union, get_args, get_origin

from eidosian_core import eidosian
from mcp.server.fastmcp import FastMCP

from .config.runtime import load_runtime_config
from .logging_utils import log_resource_read, log_tool_call

try:
    from .state import gis as _gis
except Exception:  # pragma: no cover - defensive import fallback
    _gis = None

_RUNTIME_CONFIG = load_runtime_config(gis=_gis)
mcp = FastMCP(
    "Eidosian Nexus",
    host=_RUNTIME_CONFIG.host,
    port=_RUNTIME_CONFIG.port,
    streamable_http_path=_RUNTIME_CONFIG.mount_path,
    stateless_http=_RUNTIME_CONFIG.stateless_http,
)

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


@eidosian()
def register_tool_metadata(name: str, description: str, parameters: Dict[str, Any], func: Any = None) -> None:
    _TOOL_REGISTRY[name] = {
        "name": name,
        "description": description,
        "parameters": parameters,
        "func": func,
    }


@eidosian()
def list_tool_metadata() -> List[Dict[str, Any]]:
    return list(_TOOL_REGISTRY.values())


@eidosian()
def register_resource_metadata(uri: str, description: str) -> None:
    _RESOURCE_REGISTRY[uri] = {"uri": uri, "description": description}


@eidosian()
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


_RATE_LIMIT_GLOBAL_PER_MIN = _RUNTIME_CONFIG.rate_limit_global_per_min
_RATE_LIMIT_PER_TOOL_PER_MIN = _RUNTIME_CONFIG.rate_limit_per_tool_per_min
_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_GLOBAL_TS: deque[float] = deque()
_RATE_LIMIT_TOOL_TS: dict[str, deque[float]] = defaultdict(deque)


def _enforce_rate_limit(tool_name: str) -> None:
    if _RATE_LIMIT_GLOBAL_PER_MIN <= 0 and _RATE_LIMIT_PER_TOOL_PER_MIN <= 0:
        return
    now = time.time()
    window_start = now - 60.0
    with _RATE_LIMIT_LOCK:
        while _RATE_LIMIT_GLOBAL_TS and _RATE_LIMIT_GLOBAL_TS[0] < window_start:
            _RATE_LIMIT_GLOBAL_TS.popleft()
        tool_bucket = _RATE_LIMIT_TOOL_TS[tool_name]
        while tool_bucket and tool_bucket[0] < window_start:
            tool_bucket.popleft()

        if _RATE_LIMIT_GLOBAL_PER_MIN > 0 and len(_RATE_LIMIT_GLOBAL_TS) >= _RATE_LIMIT_GLOBAL_PER_MIN:
            raise RuntimeError("MCP global rate limit exceeded (60s window)")
        if _RATE_LIMIT_PER_TOOL_PER_MIN > 0 and len(tool_bucket) >= _RATE_LIMIT_PER_TOOL_PER_MIN:
            raise RuntimeError(f"MCP tool rate limit exceeded for {tool_name} (60s window)")

        _RATE_LIMIT_GLOBAL_TS.append(now)
        tool_bucket.append(now)


@eidosian()
def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
):
    """
    Eidosian tool decorator.
    Registers tool metadata and wraps with logging/tracing.
    """

    @eidosian()
    def decorator(func):
        tool_name = name or func.__name__
        desc = (description or func.__doc__ or "").strip()
        params = parameters or _infer_parameters(func)
        register_tool_metadata(tool_name, desc, params, func=func)

        if inspect.iscoroutinefunction(func):

            @eidosian()
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
                    _enforce_rate_limit(tool_name)
                    result = await func(*args, **kwargs)
                    log_tool_call(tool_name, log_args, result, start_time=start)
                    return result
                except Exception as e:
                    log_tool_call(tool_name, log_args, None, error=str(e), start_time=start)
                    raise

        else:

            @eidosian()
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
                    _enforce_rate_limit(tool_name)
                    result = func(*args, **kwargs)
                    log_tool_call(tool_name, log_args, result, start_time=start)
                    return result
                except Exception as e:
                    log_tool_call(tool_name, log_args, None, error=str(e), start_time=start)
                    raise

        # Compatibility shim for mocks and FastMCP
        try:
            return mcp.tool(name=tool_name, description=desc)(wrapper)
        except TypeError:
            try:
                return mcp.tool(tool_name)(wrapper)
            except TypeError:
                return mcp.tool()(wrapper)

    return decorator


@eidosian()
def resource(uri: str, description: Optional[str] = None):
    @eidosian()
    def decorator(func):
        desc = (description or func.__doc__ or "").strip()
        register_resource_metadata(uri, desc)

        if inspect.iscoroutinefunction(func):

            @eidosian()
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

            @eidosian()
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
