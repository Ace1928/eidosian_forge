import asyncio
import sys
import os
from eidosian_core import eidosian

# Ensure we can find the mcp package
sys.path.append(os.path.join(os.environ["HOME"], "eidosian_forge", "eidosian_venv", "lib", "python3.12", "site-packages"))

from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

@eidosian()
async def check_health():
    url = "http://127.0.0.1:8928/sse"
    print(f"Attempting to connect to {url}...")
    try:
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                print(f"SUCCESS: Connected and retrieved {len(tools.tools)} tools.")
                
                # Check for specific critical tools
                tool_names = [t.name for t in tools.tools]
                required = ["agent_run_task", "run_shell_command"]
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
