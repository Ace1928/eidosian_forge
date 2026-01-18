try:
    from mcp.server.fastmcp import FastMCP

    print("FastMCP found")
except ImportError:
    print("FastMCP NOT found")
    import mcp

    print(dir(mcp))
