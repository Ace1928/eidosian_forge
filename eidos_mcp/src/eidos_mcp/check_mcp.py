from __future__ import annotations

import argparse
import json
import sys

from . import eidos_mcp_server  # noqa: F401
from .core import list_tool_metadata, list_resource_metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Eidos MCP wiring.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    tools = list_tool_metadata()
    resources = list_resource_metadata()

    payload = {
        "tool_count": len(tools),
        "resource_count": len(resources),
        "tools": [t["name"] for t in tools],
        "resources": [r["uri"] for r in resources],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Tools: {payload['tool_count']}")
        print(f"Resources: {payload['resource_count']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
