from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import unittest

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = str((ROOT / "eidosian_venv/bin/python3") if (ROOT / "eidosian_venv/bin/python3").exists() else Path(sys.executable))
PYTHONPATH = f"{ROOT}/eidos_mcp/src:{ROOT}"
HOST = "127.0.0.1"


def _start_server(
    transport: str,
    port: int,
    *,
    stateless_http: bool,
    mount_path: str = "/mcp",
) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": PYTHONPATH,
            "EIDOS_FORGE_DIR": str(ROOT),
            "EIDOS_MCP_TRANSPORT": transport,
            "EIDOS_MCP_MOUNT_PATH": mount_path,
            "EIDOS_MCP_STATELESS_HTTP": "1" if stateless_http else "0",
            "EIDOS_MCP_ENABLE_COMPAT_HEADERS": "1",
            "EIDOS_MCP_ENABLE_SESSION_RECOVERY": "1",
            "EIDOS_MCP_ENABLE_ERROR_RESPONSE_COMPAT": "1",
            "FASTMCP_HOST": HOST,
            "FASTMCP_PORT": str(port),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return subprocess.Popen(
        [VENV_PYTHON, "-u", "-c", "import eidos_mcp.eidos_mcp_server as s; s.main()"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_health(port: int, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    url = f"http://{HOST}:{port}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    payload = json.loads(resp.read().decode("utf-8"))
                    if payload.get("status") == "ok":
                        return
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"MCP server at {url} did not become healthy in time")


def _stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


async def _call_tool_text(session: ClientSession, name: str, arguments: dict | None = None) -> str | None:
    result = await session.call_tool(name, arguments=arguments or {})
    if result.structuredContent and "result" in result.structuredContent:
        return result.structuredContent["result"]
    if result.content:
        for content in result.content:
            if getattr(content, "type", None) == "text":
                return content.text
    return None


class TestDiagnosticsTransportMatrix(unittest.IsolatedAsyncioTestCase):
    async def test_diagnostics_ping_over_stdio(self) -> None:
        params = StdioServerParameters(
            command=VENV_PYTHON,
            args=["-u", "-c", "import eidos_mcp.eidos_mcp_server as s; s.main()"],
            env={
                **os.environ,
                "PYTHONPATH": PYTHONPATH,
                "EIDOS_FORGE_DIR": str(ROOT),
                "EIDOS_MCP_TRANSPORT": "stdio",
            },
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                ping = await _call_tool_text(session, "diagnostics_ping")
                self.assertEqual("ok", ping)

    async def test_diagnostics_ping_over_sse(self) -> None:
        port = 8931
        proc = _start_server("sse", port, stateless_http=False, mount_path="/")
        try:
            _wait_for_health(port)
            async with sse_client(f"http://{HOST}:{port}/sse") as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    ping = await _call_tool_text(session, "diagnostics_ping")
                    self.assertEqual("ok", ping)
        finally:
            _stop_server(proc)

    async def test_diagnostics_ping_over_streamable_http_stateful(self) -> None:
        port = 8932
        proc = _start_server("streamable-http", port, stateless_http=False)
        try:
            _wait_for_health(port)
            async with streamable_http_client(f"http://{HOST}:{port}/mcp") as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    ping = await _call_tool_text(session, "diagnostics_ping")
                    self.assertEqual("ok", ping)
        finally:
            _stop_server(proc)

    async def test_diagnostics_ping_over_streamable_http_stateless(self) -> None:
        port = 8933
        proc = _start_server("streamable-http", port, stateless_http=True)
        try:
            _wait_for_health(port)
            async with streamable_http_client(f"http://{HOST}:{port}/mcp") as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    ping = await _call_tool_text(session, "diagnostics_ping")
                    self.assertEqual("ok", ping)
        finally:
            _stop_server(proc)
