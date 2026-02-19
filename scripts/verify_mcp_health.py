import asyncio
import sys
import os
from pathlib import Path
import urllib.request
import urllib.error

# Ensure local repo modules are available when not installed globally.
FORGE_ROOT = Path(__file__).resolve().parent.parent
for extra_path in (FORGE_ROOT / "lib", FORGE_ROOT / "eidos_mcp" / "src", FORGE_ROOT):
    str_path = str(extra_path)
    if extra_path.exists() and str_path not in sys.path:
        sys.path.insert(0, str_path)

from eidosian_core import eidosian
from eidosian_core.ports import get_service_url

from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mcp.client.session import ClientSession


def _check_http_health(base_url: str) -> None:
    health_url = base_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=3) as response:
            if response.status != 200:
                raise RuntimeError(f"Unexpected health status: {response.status}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Health endpoint unavailable: {health_url}") from exc

@eidosian()
async def check_health():
    default_mcp_url = get_service_url("eidos_mcp", default_port=8928, default_path="/mcp")
    url = os.environ.get("EIDOS_MCP_URL", default_mcp_url)
    print(f"Attempting to connect to {url}...")
    base_url = url.replace("/mcp", "") if "/mcp" in url else url
    _check_http_health(base_url)
    try:
        client = streamable_http_client if "/mcp" in url else sse_client
        async with client(url) as (read, write, *_):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                print(f"SUCCESS: Connected and retrieved {len(tools.tools)} tools.")
                
                # Check for specific critical tools
                tool_names = [t.name for t in tools.tools]
                required = ["agent_run_task", "run_shell_command", "memory_stats", "kb_search"]
                for r in required:
                    if r in tool_names:
                        print(f"  [+] Found required tool: {r}")
                    else:
                        print(f"  [-] MISSING tool: {r}")
                        
    except Exception as e:
        print(f"FAILURE: Could not connect to MCP server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(check_health())
