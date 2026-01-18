#!/usr/bin/env python3
"""
💎 Eidosian Resource Fetcher
Connects to the local Eidosian Nexus (MCP Server) via stdio and retrieves a specific resource.
Used to dynamically inject context into agent sessions.
"""

import sys
import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def fetch_resource(uri: str):
    # Define server parameters - MUST match the run_server.sh config
    server_params = StdioServerParameters(
        command="/home/lloyd/eidosian_forge/eidos_mcp/run_server.sh",
        args=[],
        env={**os.environ, "PYTHONPATH": "/home/lloyd"},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List resources to find the matching one (optional, but good for validation)
            # resources = await session.list_resources()

            # Read the specific resource
            try:
                result = await session.read_resource(uri)
                # content is a list of TextResourceContents or BlobResourceContents
                for content in result.contents:
                    print(content.text)
            except Exception as e:
                print(f"Error fetching {uri}: {e}", file=sys.stderr)
                sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: eidos_fetch.py <resource_uri>", file=sys.stderr)
        print("Example: eidos_fetch.py eidos://persona", file=sys.stderr)
        sys.exit(1)

    uri = sys.argv[1]
    try:
        asyncio.run(fetch_resource(uri))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
