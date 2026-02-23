from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from typing import Any

from eidosian_core import eidosian

from . import eidos_mcp_server  # noqa: F401
from .core import list_resource_metadata, list_tool_metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Eidos MCP wiring.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Build streamable-http ASGI app to verify server bootstrap wiring.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if any diagnostic check fails.",
    )
    parser.add_argument(
        "--expect-env",
        action="append",
        default=[],
        help="Environment variable expected to be non-empty (repeatable).",
    )
    return parser


def _check_import(path: str, attr: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(path)
        ok = hasattr(module, attr)
        return {"ok": bool(ok), "target": f"{path}.{attr}", "detail": None if ok else "missing attribute"}
    except Exception as exc:
        return {"ok": False, "target": f"{path}.{attr}", "detail": str(exc)}


def _component_checks(expected_env: list[str], smoke_test: bool) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    try:
        checks["mcp_version"] = {"ok": True, "value": importlib.metadata.version("mcp")}
    except Exception as exc:
        checks["mcp_version"] = {"ok": False, "detail": str(exc)}

    checks["mcp_client_session"] = _check_import("mcp", "ClientSession")
    checks["mcp_stdio_params"] = _check_import("mcp", "StdioServerParameters")
    checks["mcp_stdio_client"] = _check_import("mcp.client.stdio", "stdio_client")

    recommended_env = {
        "EIDOS_HOME_DIR": bool(os.environ.get("EIDOS_HOME_DIR")),
        "EIDOS_FORGE_DIR": bool(os.environ.get("EIDOS_FORGE_DIR")),
    }
    required_names = sorted(set(name for name in expected_env if name))
    required_status = {name: bool(os.environ.get(name)) for name in required_names}
    checks["recommended_env"] = {
        "ok": True,
        "values": recommended_env,
    }
    checks["expected_env"] = {
        "ok": all(required_status.values()) if required_status else True,
        "values": required_status,
    }

    if smoke_test:
        try:
            app = eidos_mcp_server._build_streamable_http_app(eidos_mcp_server._runtime_config().mount_path)  # noqa: SLF001
            checks["server_smoke_build"] = {
                "ok": app is not None,
                "detail": "ASGI app created",
            }
        except Exception as exc:
            checks["server_smoke_build"] = {"ok": False, "detail": str(exc)}
    else:
        checks["server_smoke_build"] = {"ok": True, "detail": "skipped"}

    return checks


@eidosian()
def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    tools = list_tool_metadata()
    resources = list_resource_metadata()
    checks = _component_checks(args.expect_env, bool(args.smoke_test))

    payload = {
        "tool_count": len(tools),
        "resource_count": len(resources),
        "tools": [t["name"] for t in tools],
        "resources": [r["uri"] for r in resources],
        "component_checks": checks,
    }
    has_failures = any(not bool(rec.get("ok")) for rec in checks.values() if isinstance(rec, dict))

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Tools: {payload['tool_count']}")
        print(f"Resources: {payload['resource_count']}")
        print(f"Checks: {'OK' if not has_failures else 'FAILED'}")
        if has_failures:
            for name, rec in checks.items():
                if isinstance(rec, dict) and not rec.get("ok"):
                    print(f"- {name}: {rec.get('detail') or rec}")

    if args.strict and has_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
