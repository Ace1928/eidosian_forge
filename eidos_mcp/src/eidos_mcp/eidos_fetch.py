"""Fetch MCP resources via HTTP (Streamable HTTP transport)."""
from __future__ import annotations

import argparse
import asyncio
from typing import Optional

import httpx
from eidosian_core import eidosian


MCP_URL = "http://localhost:8928/mcp"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch an MCP resource via HTTP.")
    parser.add_argument("uri", help="Resource URI (e.g. eidos://persona)")
    parser.add_argument(
        "--url",
        default=MCP_URL,
        help="MCP server URL.",
    )
    return parser


async def _fetch(uri: str, url: str) -> str:
    """Fetch a resource from the MCP server via Streamable HTTP."""
    async with httpx.AsyncClient() as client:
        # Initialize session
        init_resp = await client.post(
            url,
            json={"jsonrpc": "2.0", "method": "initialize", "id": 1, "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "eidos_fetch", "version": "1.0.0"}
            }},
            headers={"Content-Type": "application/json"},
        )
        init_resp.raise_for_status()
        
        # Read resource
        read_resp = await client.post(
            url,
            json={"jsonrpc": "2.0", "method": "resources/read", "id": 2, "params": {"uri": uri}},
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
    content = asyncio.run(_fetch(args.uri, args.url))
    if content:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
