from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch an MCP resource via stdio.")
    parser.add_argument("uri", help="Resource URI (e.g. eidos://persona)")
    parser.add_argument(
        "--python",
        default=os.environ.get("EIDOS_PYTHON_BIN", sys.executable),
        help="Python executable to launch the MCP server.",
    )
    return parser


async def _fetch(uri: str, python_bin: str) -> str:
    server_args = ["-m", "eidos_mcp.eidos_mcp_server"]
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH", "")
    forge_dir = os.environ.get("EIDOS_FORGE_DIR", "/home/lloyd/eidosian_forge")
    mcp_src = os.path.join(forge_dir, "eidos_mcp", "src")
    
    paths_to_add = [mcp_src, forge_dir]
    for p in paths_to_add:
        if p not in pythonpath:
             pythonpath = f"{p}:{pythonpath}" if pythonpath else p
    
    env["PYTHONPATH"] = pythonpath

    params = StdioServerParameters(command=python_bin, args=server_args, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.read_resource(uri)
            if result.contents:
                return result.contents[0].text
    return ""


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    content = asyncio.run(_fetch(args.uri, args.python))
    if content:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
