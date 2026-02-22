from __future__ import annotations

from typing import Any

from eidos_mcp.config.runtime import DEFAULT_ALLOWED_ORIGINS, load_runtime_config


class _FakeGIS:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def get(self, key: str, default: Any = None, use_env: bool = False) -> Any:
        del use_env
        current: Any = self._payload
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current


def test_runtime_config_reads_gis_values() -> None:
    gis = _FakeGIS(
        {
            "mcp": {
                "transport": "stdio",
                "mount_path": "custom-mcp",
                "stateless_http": True,
                "enable_compat_headers": False,
                "enable_session_recovery": False,
                "enable_error_response_compat": False,
                "enforce_origin": False,
                "audit_transport": False,
                "allowed_origins": ["https://example.com"],
                "host": "0.0.0.0",
                "port": 9123,
                "log_level": "DEBUG",
                "reload": True,
                "rate_limit_global_per_min": 77,
                "rate_limit_per_tool_per_min": 55,
                "oauth": {
                    "provider": "google",
                    "audience": "aud-1",
                    "static_bearer": "secret-token",
                },
            }
        }
    )
    cfg = load_runtime_config(gis=gis, environ={})
    assert cfg.transport == "stdio"
    assert cfg.mount_path == "/custom-mcp"
    assert cfg.stateless_http is True
    assert cfg.enable_compat_headers is False
    assert cfg.enable_session_recovery is False
    assert cfg.enable_error_response_compat is False
    assert cfg.enforce_origin is False
    assert cfg.audit_transport is False
    assert cfg.allowed_origins == ("https://example.com",)
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 9123
    assert cfg.log_level == "debug"
    assert cfg.reload is True
    assert cfg.rate_limit_global_per_min == 77
    assert cfg.rate_limit_per_tool_per_min == 55
    assert cfg.oauth_provider == "google"
    assert cfg.oauth_audience == "aud-1"
    assert cfg.oauth_static_bearer == "secret-token"


def test_runtime_config_env_overrides_gis() -> None:
    gis = _FakeGIS({"mcp": {"transport": "stdio", "host": "0.0.0.0", "port": 9123}})
    env = {
        "EIDOS_MCP_TRANSPORT": "streamable-http",
        "FASTMCP_HOST": "127.0.0.1",
        "FASTMCP_PORT": "8930",
        "EIDOS_MCP_ALLOWED_ORIGINS": "http://localhost:8080, https://example.org",
        "EIDOS_MCP_ENABLE_COMPAT_HEADERS": "1",
        "EIDOS_MCP_ENABLE_SESSION_RECOVERY": "1",
        "EIDOS_MCP_ENFORCE_ORIGIN": "1",
    }
    cfg = load_runtime_config(gis=gis, environ=env)
    assert cfg.transport == "streamable-http"
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8930
    assert cfg.allowed_origins == ("http://localhost:8080", "https://example.org")


def test_runtime_config_defaults_are_stable() -> None:
    cfg = load_runtime_config(gis=None, environ={})
    assert cfg.transport == "streamable-http"
    assert cfg.mount_path == "/mcp"
    assert cfg.allowed_origins == DEFAULT_ALLOWED_ORIGINS
    assert cfg.rate_limit_global_per_min >= 0
    assert cfg.rate_limit_per_tool_per_min >= 0
