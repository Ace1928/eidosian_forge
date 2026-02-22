from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from eidosian_core.ports import get_service_port

DEFAULT_ALLOWED_ORIGINS: tuple[str, ...] = (
    "http://127.0.0.1",
    "http://localhost",
    "http://[::1]",
    "https://127.0.0.1",
    "https://localhost",
    "https://[::1]",
)


@dataclass(frozen=True)
class RuntimeConfig:
    transport: str = "streamable-http"
    mount_path: str = "/mcp"
    stateless_http: bool = False
    enable_compat_headers: bool = True
    enable_session_recovery: bool = True
    enable_error_response_compat: bool = True
    enforce_origin: bool = True
    audit_transport: bool = True
    allowed_origins: tuple[str, ...] = DEFAULT_ALLOWED_ORIGINS
    host: str = "127.0.0.1"
    port: int = 8928
    log_level: str = "info"
    reload: bool = False
    rate_limit_global_per_min: int = 600
    rate_limit_per_tool_per_min: int = 300
    oauth_provider: str = ""
    oauth_audience: Optional[str] = None
    oauth_static_bearer: Optional[str] = None


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_str(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value)


def _coerce_origins(value: Any) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_ALLOWED_ORIGINS
    if isinstance(value, (list, tuple, set)):
        normalized = tuple(str(v).strip().lower() for v in value if str(v).strip())
        return normalized or DEFAULT_ALLOWED_ORIGINS
    raw = str(value).strip()
    if not raw:
        return DEFAULT_ALLOWED_ORIGINS
    normalized = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    return normalized or DEFAULT_ALLOWED_ORIGINS


def _gis_get(gis: Any, key: str, default: Any = None) -> Any:
    if gis is None or not hasattr(gis, "get"):
        return default
    try:
        return gis.get(key, default=default, use_env=False)
    except TypeError:
        try:
            return gis.get(key, default=default)
        except TypeError:
            return gis.get(key)
    except Exception:
        return default


def _resolve_value(
    *,
    env_keys: Sequence[str],
    gis_key: str,
    default: Any,
    environ: Mapping[str, str],
    gis: Any = None,
) -> Any:
    for env_key in env_keys:
        value = environ.get(env_key)
        if value is not None and value != "":
            return value
    gis_value = _gis_get(gis, gis_key, default=None)
    if gis_value is not None and gis_value != "":
        return gis_value
    return default


def load_runtime_config(*, gis: Any = None, environ: Optional[Mapping[str, str]] = None) -> RuntimeConfig:
    env = os.environ if environ is None else environ

    transport = _coerce_str(
        _resolve_value(
            env_keys=("EIDOS_MCP_TRANSPORT",),
            gis_key="mcp.transport",
            default="streamable-http",
            environ=env,
            gis=gis,
        ),
        default="streamable-http",
    )
    mount_path = _coerce_str(
        _resolve_value(
            env_keys=("EIDOS_MCP_MOUNT_PATH", "FASTMCP_STREAMABLE_HTTP_PATH"),
            gis_key="mcp.mount_path",
            default="/mcp",
            environ=env,
            gis=gis,
        ),
        default="/mcp",
    )
    if not mount_path.startswith("/"):
        mount_path = f"/{mount_path}"

    stateless_http = _coerce_bool(
        _resolve_value(
            env_keys=("EIDOS_MCP_STATELESS_HTTP",),
            gis_key="mcp.stateless_http",
            default=False,
            environ=env,
            gis=gis,
        ),
        default=False,
    )
    enable_compat_headers = _coerce_bool(
        _resolve_value(
            env_keys=("EIDOS_MCP_ENABLE_COMPAT_HEADERS",),
            gis_key="mcp.enable_compat_headers",
            default=True,
            environ=env,
            gis=gis,
        ),
        default=True,
    )
    enable_session_recovery = _coerce_bool(
        _resolve_value(
            env_keys=("EIDOS_MCP_ENABLE_SESSION_RECOVERY",),
            gis_key="mcp.enable_session_recovery",
            default=True,
            environ=env,
            gis=gis,
        ),
        default=True,
    )
    enable_error_response_compat = _coerce_bool(
        _resolve_value(
            env_keys=("EIDOS_MCP_ENABLE_ERROR_RESPONSE_COMPAT",),
            gis_key="mcp.enable_error_response_compat",
            default=True,
            environ=env,
            gis=gis,
        ),
        default=True,
    )
    enforce_origin = _coerce_bool(
        _resolve_value(
            env_keys=("EIDOS_MCP_ENFORCE_ORIGIN",),
            gis_key="mcp.enforce_origin",
            default=True,
            environ=env,
            gis=gis,
        ),
        default=True,
    )
    audit_transport = _coerce_bool(
        _resolve_value(
            env_keys=("EIDOS_MCP_AUDIT_TRANSPORT",),
            gis_key="mcp.audit_transport",
            default=True,
            environ=env,
            gis=gis,
        ),
        default=True,
    )
    allowed_origins = _coerce_origins(
        _resolve_value(
            env_keys=("EIDOS_MCP_ALLOWED_ORIGINS",),
            gis_key="mcp.allowed_origins",
            default=None,
            environ=env,
            gis=gis,
        )
    )

    host = _coerce_str(
        _resolve_value(
            env_keys=("FASTMCP_HOST", "EIDOS_MCP_HOST"),
            gis_key="mcp.host",
            default="127.0.0.1",
            environ=env,
            gis=gis,
        ),
        default="127.0.0.1",
    )
    configured_port = _resolve_value(
        env_keys=("FASTMCP_PORT", "EIDOS_MCP_PORT"),
        gis_key="mcp.port",
        default=None,
        environ=env,
        gis=gis,
    )
    if configured_port is None:
        port = get_service_port(
            "eidos_mcp",
            default=8928,
            env_keys=("FASTMCP_PORT", "EIDOS_MCP_PORT"),
        )
    else:
        port = _coerce_int(configured_port, default=8928)
    log_level = _coerce_str(
        _resolve_value(
            env_keys=("FASTMCP_LOG_LEVEL",),
            gis_key="mcp.log_level",
            default="info",
            environ=env,
            gis=gis,
        ),
        default="info",
    ).lower()
    reload = _coerce_bool(
        _resolve_value(
            env_keys=("FASTMCP_RELOAD", "EIDOS_MCP_RELOAD"),
            gis_key="mcp.reload",
            default=False,
            environ=env,
            gis=gis,
        ),
        default=False,
    )
    rate_limit_global = max(
        0,
        _coerce_int(
            _resolve_value(
                env_keys=("EIDOS_MCP_RATE_LIMIT_GLOBAL_PER_MIN",),
                gis_key="mcp.rate_limit_global_per_min",
                default=600,
                environ=env,
                gis=gis,
            ),
            default=600,
        ),
    )
    rate_limit_per_tool = max(
        0,
        _coerce_int(
            _resolve_value(
                env_keys=("EIDOS_MCP_RATE_LIMIT_PER_TOOL_PER_MIN",),
                gis_key="mcp.rate_limit_per_tool_per_min",
                default=300,
                environ=env,
                gis=gis,
            ),
            default=300,
        ),
    )

    oauth_provider = _coerce_str(
        _resolve_value(
            env_keys=("EIDOS_OAUTH2_PROVIDER",),
            gis_key="mcp.oauth.provider",
            default="",
            environ=env,
            gis=gis,
        ),
        default="",
    ).lower()
    oauth_audience = _resolve_value(
        env_keys=("EIDOS_OAUTH2_AUDIENCE",),
        gis_key="mcp.oauth.audience",
        default=None,
        environ=env,
        gis=gis,
    )
    oauth_static_bearer = _resolve_value(
        env_keys=("EIDOS_OAUTH2_STATIC_BEARER",),
        gis_key="mcp.oauth.static_bearer",
        default=None,
        environ=env,
        gis=gis,
    )

    return RuntimeConfig(
        transport=transport,
        mount_path=mount_path,
        stateless_http=stateless_http,
        enable_compat_headers=enable_compat_headers,
        enable_session_recovery=enable_session_recovery,
        enable_error_response_compat=enable_error_response_compat,
        enforce_origin=enforce_origin,
        audit_transport=audit_transport,
        allowed_origins=allowed_origins,
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
        rate_limit_global_per_min=rate_limit_global,
        rate_limit_per_tool_per_min=rate_limit_per_tool,
        oauth_provider=oauth_provider,
        oauth_audience=oauth_audience if oauth_audience else None,
        oauth_static_bearer=oauth_static_bearer if oauth_static_bearer else None,
    )
