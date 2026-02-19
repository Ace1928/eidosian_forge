"""Fetch MCP resources via stdio or HTTP."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import httpx
except Exception:  # pragma: no cover - optional runtime dependency
    httpx = None

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from . import FORGE_ROOT
from eidosian_core import eidosian
from eidosian_core.ports import get_service_url


MCP_URL = get_service_url("eidos_mcp", default_port=8928, default_host="localhost", default_path="/mcp")


def _build_parser() -> argparse.ArgumentParser:
    default_python = os.environ.get(
        "EIDOS_PYTHON_BIN", str(FORGE_ROOT / "eidosian_venv" / "bin" / "python3")
    )
    if not Path(default_python).exists():
        default_python = sys.executable

    parser = argparse.ArgumentParser(description="Fetch an MCP resource.")
    parser.add_argument("uri", help="Resource URI (e.g. eidos://persona)")
    parser.add_argument(
        "--transport",
        choices=["auto", "stdio", "http"],
        default=os.environ.get("EIDOS_FETCH_TRANSPORT", "auto"),
        help="Transport mode (default: auto)",
    )
    parser.add_argument(
        "--python",
        default=default_python,
        help="Python executable to launch the MCP server in stdio mode.",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("EIDOS_MCP_URL", MCP_URL),
        help="MCP server URL for HTTP mode.",
    )
    return parser


def _augment_pythonpath(env: dict[str, str]) -> None:
    pythonpath = env.get("PYTHONPATH", "")
    forge_dir = os.environ.get("EIDOS_FORGE_DIR", str(FORGE_ROOT))
    mcp_src = os.path.join(forge_dir, "eidos_mcp", "src")
    lib_path = os.path.join(forge_dir, "lib")
    paths_to_add = [mcp_src, forge_dir, lib_path]
    for path in paths_to_add:
        if path not in pythonpath:
            pythonpath = f"{path}:{pythonpath}" if pythonpath else path
    env["PYTHONPATH"] = pythonpath


async def _fetch_stdio(uri: str, python_bin: str) -> str:
    server_args = ["-m", "eidos_mcp.eidos_mcp_server"]
    env = dict(os.environ)
    _augment_pythonpath(env)
    # Force short-lived stdio transport even if the caller environment sets HTTP transport.
    env["EIDOS_MCP_TRANSPORT"] = "stdio"
    env.pop("EIDOS_MCP_MOUNT_PATH", None)
    env.setdefault("EIDOS_GIS_PATH", str(Path.home() / ".eidosian" / "gis_data.local.json"))

    params = StdioServerParameters(command=python_bin, args=server_args, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.read_resource(uri)
            if result.contents:
                return result.contents[0].text
    return ""


async def _fetch_http(uri: str, url: str) -> str:
    if httpx is None:
        raise RuntimeError("httpx is not installed for HTTP transport")
    async with httpx.AsyncClient(timeout=8.0) as client:
        init_resp = await client.post(
            url,
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "id": 1,
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "eidos_fetch", "version": "1.0.0"},
                },
            },
            headers={"Content-Type": "application/json"},
        )
        init_resp.raise_for_status()

        read_resp = await client.post(
            url,
            json={
                "jsonrpc": "2.0",
                "method": "resources/read",
                "id": 2,
                "params": {"uri": uri},
            },
            headers={"Content-Type": "application/json"},
        )
        read_resp.raise_for_status()
        data = read_resp.json()
        if "result" in data and "contents" in data["result"]:
            contents = data["result"]["contents"]
            if contents and "text" in contents[0]:
                return contents[0]["text"]
    return ""


@eidosian()
def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.transport == "stdio":
            content = asyncio.run(_fetch_stdio(args.uri, args.python))
        elif args.transport == "http":
            content = asyncio.run(_fetch_http(args.uri, args.url))
        else:
            # Auto mode: stdio first for self-contained portability, HTTP fallback.
            try:
                content = asyncio.run(_fetch_stdio(args.uri, args.python))
            except Exception:
                content = asyncio.run(_fetch_http(args.uri, args.url))
    except Exception as exc:
        print(f"eidos_fetch error: {exc}", file=sys.stderr)
        return 1

    if content:
        print(content)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
