from __future__ import annotations

import asyncio
import os
import sys
import unittest
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = str((ROOT / "eidosian_venv/bin/python3") if (ROOT / "eidosian_venv/bin/python3").exists() else Path(sys.executable))
PYTHONPATH = f"{ROOT}/eidos_mcp/src:{ROOT}"


async def _call_tool(
    session: ClientSession,
    name: str,
    arguments: dict | None = None,
    timeout_sec: float = 30.0,
) -> str:
    result = await asyncio.wait_for(
        session.call_tool(name, arguments=arguments or {}),
        timeout=timeout_sec,
    )
    if result.structuredContent and "result" in result.structuredContent:
        return str(result.structuredContent["result"])
    if result.content:
        for content in result.content:
            if getattr(content, "type", None) == "text":
                return content.text
    return ""


@asynccontextmanager
async def _stdio_session() -> AsyncIterator[ClientSession]:
    params = StdioServerParameters(
        command=VENV_PYTHON,
        args=["-u", "-c", "import eidos_mcp.eidos_mcp_server as s; s.main()"],
        env={
            **os.environ,
            "PYTHONPATH": PYTHONPATH,
            "EIDOS_FORGE_DIR": str(ROOT),
            "EIDOS_MCP_TRANSPORT": "stdio",
            "EIDOS_MCP_STATELESS_HTTP": "1",
        },
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


ToolValidator = Callable[[str], bool]
TOOL_CASES: list[tuple[str, dict, ToolValidator, str]] = [
    ("system_info", {}, lambda r: "Linux" in r, "system_info missing platform"),
    ("diagnostics_ping", {}, lambda r: r.strip() == "ok", "diagnostics_ping did not return ok"),
    ("diagnostics_metrics", {}, lambda r: r.strip().startswith("{"), "diagnostics_metrics did not return JSON"),
    ("memory_stats", {}, lambda r: "count" in r or "episodic_count" in r, "memory_stats missing count"),
    ("transaction_list", {"limit": 3}, lambda r: "timestamp" in r or r.strip() in {"[]", ""}, "transaction_list unexpected shape"),
    ("gis_snapshot", {}, lambda r: "Snapshot created" in r, "gis_snapshot failed"),
    ("type_snapshot", {}, lambda r: "Snapshot created" in r, "type_snapshot failed"),
    ("tiered_memory_stats", {}, lambda r: "tier" in r or "count" in r, "tiered_memory_stats missing expected fields"),
    ("tika_cache_stats", {}, lambda r: "hits" in r or "entries" in r or "{" in r, "tika_cache_stats missing expected fields"),
    ("moltbook_status", {}, lambda r: "\"success\"" in r or "\"ok\"" in r, "moltbook_status missing expected keys"),
    ("moltbook_me", {}, lambda r: "\"agent\"" in r or "\"ok\"" in r, "moltbook_me missing expected keys"),
    ("moltbook_feed", {"sort": "new", "limit": 3}, lambda r: "\"posts\"" in r or "\"ok\"" in r, "moltbook_feed missing expected keys"),
    ("moltbook_posts", {"sort": "hot", "limit": 3}, lambda r: "\"posts\"" in r or "\"ok\"" in r, "moltbook_posts missing expected keys"),
    ("moltbook_dm_check", {}, lambda r: "\"has_activity\"" in r or "\"ok\"" in r, "moltbook_dm_check missing expected keys"),
    (
        "consciousness_kernel_status",
        {},
        lambda r: "\"workspace\"" in r and "\"rci\"" in r and "\"watchdog\"" in r and "\"payload_safety\"" in r,
        "consciousness_kernel_status missing expected keys",
    ),
    (
        "consciousness_bridge_status",
        {},
        lambda r: "\"memory_bridge\"" in r and "\"knowledge_bridge\"" in r,
        "consciousness_bridge_status missing expected keys",
    ),
    (
        "consciousness_kernel_benchmark",
        {"ticks": 2, "persist": False},
        lambda r: "\"benchmark_id\"" in r and "\"composite\"" in r,
        "consciousness_kernel_benchmark missing expected keys",
    ),
    (
        "consciousness_kernel_stress_benchmark",
        {
            "ticks": 2,
            "event_fanout": 4,
            "broadcast_fanout": 2,
            "payload_chars": 4000,
            "max_payload_bytes": 1024,
            "persist": False,
        },
        lambda r: "\"benchmark_id\"" in r and "\"payload_truncations_observed\"" in r,
        "consciousness_kernel_stress_benchmark missing expected keys",
    ),
    (
        "consciousness_kernel_red_team",
        {"persist": False, "max_scenarios": 1, "quick": True},
        lambda r: "\"run_id\"" in r and "\"scenario_count\"" in r and "\"pass_ratio\"" in r,
        "consciousness_kernel_red_team missing expected keys",
    ),
    (
        "consciousness_kernel_full_benchmark",
        {
            "rounds": 1,
            "bench_ticks": 2,
            "trial_ticks": 1,
            "run_mcp": False,
            "run_llm": False,
            "run_red_team": False,
            "persist": False,
        },
        lambda r: "\"benchmark_id\"" in r and "\"integrated\"" in r,
        "consciousness_kernel_full_benchmark missing expected keys",
    ),
]


class TestMcpToolCallsIndividual(unittest.IsolatedAsyncioTestCase):
    async def test_list_tools_and_resources(self) -> None:
        async with _stdio_session() as session:
            tools = await asyncio.wait_for(session.list_tools(), timeout=20)
            resources = await asyncio.wait_for(session.list_resources(), timeout=20)
        self.assertGreater(len(tools.tools), 0)
        self.assertGreater(len(resources.resources), 0)

    async def test_resource_config(self) -> None:
        async with _stdio_session() as session:
            result = await asyncio.wait_for(session.read_resource("eidos://config"), timeout=20)
        self.assertTrue(result.contents)

    async def test_resource_persona(self) -> None:
        async with _stdio_session() as session:
            result = await asyncio.wait_for(session.read_resource("eidos://persona"), timeout=20)
        self.assertTrue(result.contents)

    async def test_resource_roadmap(self) -> None:
        async with _stdio_session() as session:
            result = await asyncio.wait_for(session.read_resource("eidos://roadmap"), timeout=20)
        self.assertTrue(result.contents)

    async def test_resource_todo(self) -> None:
        async with _stdio_session() as session:
            result = await asyncio.wait_for(session.read_resource("eidos://todo"), timeout=20)
        self.assertTrue(result.contents)

    async def test_resource_consciousness_runtime_latest_full_benchmark(self) -> None:
        async with _stdio_session() as session:
            result = await asyncio.wait_for(
                session.read_resource("eidos://consciousness/runtime-latest-full-benchmark"),
                timeout=20,
            )
        self.assertTrue(result.contents)

    async def test_resource_consciousness_runtime_latest_red_team(self) -> None:
        async with _stdio_session() as session:
            result = await asyncio.wait_for(
                session.read_resource("eidos://consciousness/runtime-latest-red-team"),
                timeout=20,
            )
        self.assertTrue(result.contents)

    async def test_resource_consciousness_runtime_latest_stress_benchmark(self) -> None:
        async with _stdio_session() as session:
            result = await asyncio.wait_for(
                session.read_resource("eidos://consciousness/runtime-latest-stress-benchmark"),
                timeout=20,
            )
        self.assertTrue(result.contents)

    async def test_resource_consciousness_runtime_integrations(self) -> None:
        async with _stdio_session() as session:
            result = await asyncio.wait_for(
                session.read_resource("eidos://consciousness/runtime-integrations"),
                timeout=20,
            )
        self.assertTrue(result.contents)


def _make_tool_test(
    tool_name: str,
    arguments: dict,
    validator: ToolValidator,
    failure_reason: str,
):
    async def _test(self: TestMcpToolCallsIndividual) -> None:
        async with _stdio_session() as session:
            result = await _call_tool(session, tool_name, arguments=arguments)
        self.assertTrue(validator(result), f"{failure_reason}: {result[:400]}")

    return _test


for _tool_name, _args, _validator, _reason in TOOL_CASES:
    setattr(
        TestMcpToolCallsIndividual,
        f"test_tool_call_{_tool_name}",
        _make_tool_test(_tool_name, _args, _validator, _reason),
    )
