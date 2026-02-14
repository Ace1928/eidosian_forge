#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = "http://127.0.0.1:8928/mcp"


async def _call_tool(session: ClientSession, name: str, arguments: dict | None = None) -> str:
    result = await session.call_tool(name, arguments=arguments or {})
    if result.structuredContent and "result" in result.structuredContent:
        return str(result.structuredContent["result"])
    if result.content:
        for content in result.content:
            if getattr(content, "type", None) == "text":
                return content.text
    return ""


async def main() -> int:
    payload: dict[str, object] = {
        "url": URL,
        "ok": False,
        "tools": None,
        "diagnostics_ping": None,
        "memory_stats": None,
        "eidos_memory_stats": None,
        "error": None,
    }
    try:
        async with streamable_http_client(URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                payload["tools"] = len(tools.tools)
                payload["diagnostics_ping"] = await _call_tool(session, "diagnostics_ping")
                payload["memory_stats"] = await _call_tool(session, "memory_stats")[:500]
                payload["eidos_memory_stats"] = await _call_tool(session, "eidos_memory_stats")[:500]
                payload["ok"] = True
    except Exception as exc:
        payload["error"] = repr(exc)

    print(json.dumps(payload, ensure_ascii=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
