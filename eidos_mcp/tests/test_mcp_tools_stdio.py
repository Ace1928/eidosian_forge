import asyncio
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
import unittest
import pytest

from mcp import ClientSession, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.client.stdio import stdio_client
SANDBOX = Path("/home/lloyd/.eidosian/tmp/mcp_sandbox")
AUDIT_DATA = Path("/home/lloyd/eidosian_forge/audit_data/coverage_map.json")
TODO_PATH = Path("/home/lloyd/TODO.md")
VENV_PYTHON = "/home/lloyd/eidosian_forge/eidosian_venv/bin/python3"
MCP_HOST = "127.0.0.1"
MCP_PORT = 8928


def _start_http_server() -> subprocess.Popen:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", "/home/lloyd/eidosian_forge/eidos_mcp/src:/home/lloyd/eidosian_forge")
    env["EIDOS_FORGE_DIR"] = "/home/lloyd/eidosian_forge"
    env["EIDOS_MCP_TRANSPORT"] = "streamable-http"
    env["FASTMCP_HOST"] = MCP_HOST
    env["FASTMCP_PORT"] = str(MCP_PORT)
    env["PYTHONUNBUFFERED"] = "1"
    env["FASTMCP_LOG_LEVEL"] = "DEBUG"
    return subprocess.Popen(
        [VENV_PYTHON, "-u", "-c", "import eidos_mcp.eidos_mcp_server as s; s.main()"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_health(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    url = f"http://{MCP_HOST}:{MCP_PORT}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            stdout = proc.stdout.read() if proc.stdout else ""
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"MCP HTTP server exited early.\nstdout:\n{stdout}\nstderr:\n{stderr}")
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.3)
    stdout = proc.stdout.read() if proc.stdout else ""
    stderr = proc.stderr.read() if proc.stderr else ""
    raise RuntimeError(
        "MCP HTTP server did not become healthy in time"
        f"\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )


def _extract_txn_id(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"\(([a-f0-9\-]+)\)", text)
    return match.group(1) if match else None


def _extract_node_id(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"Added node: ([a-f0-9\-]+)", text)
    return match.group(1) if match else None


def _backup_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _restore_file(path: Path, content: str | None) -> None:
    if content is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


async def _call_tool(session: ClientSession, name: str, arguments: dict | None = None) -> str | None:
    result = await session.call_tool(name, arguments=arguments or {})
    if result.structuredContent and "result" in result.structuredContent:
        return result.structuredContent["result"]
    if result.content:
        for content in result.content:
            if getattr(content, "type", None) == "text":
                return content.text
    return None


class TestMcpToolsStdio(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.todo_backup = _backup_file(TODO_PATH)
        self.audit_backup = _backup_file(AUDIT_DATA)

    async def asyncTearDown(self) -> None:
        _restore_file(TODO_PATH, self.todo_backup)
        _restore_file(AUDIT_DATA, self.audit_backup)
        if SANDBOX.exists():
            shutil.rmtree(SANDBOX)

    async def _run_tool_flow(self, session: ClientSession) -> None:
        tools = await session.list_tools()
        resources = await session.list_resources()
        self.assertGreater(len(tools.tools), 0)
        self.assertGreater(len(resources.resources), 0)

        self.assertIsNotNone(await _call_tool(session, "mcp_list_tools"))
        self.assertIsNotNone(await _call_tool(session, "mcp_list_resources"))

        for uri in ["eidos://config", "eidos://persona", "eidos://roadmap", "eidos://todo"]:
            result = await session.read_resource(uri)
            self.assertTrue(result.contents)

        self.assertIn("Linux", await _call_tool(session, "system_info"))
        self.assertEqual("ok", await _call_tool(session, "diagnostics_ping"))

        sandbox_file = SANDBOX / "sample.txt"
        sandbox_file_2 = SANDBOX / "sample_copy.txt"

        create_result = await _call_tool(session, "file_create", {"file_path": str(sandbox_file)})
        self.assertTrue(
            create_result.startswith("Committed") or create_result == "No-op: Path already exists"
        )
        write_result = await _call_tool(
            session,
            "file_write",
            {"file_path": str(sandbox_file), "content": "Eidosian MCP sandbox"},
        )
        self.assertIn("Committed", write_result)
        read_result = await _call_tool(session, "file_read", {"file_path": str(sandbox_file)})
        self.assertEqual("Eidosian MCP sandbox", read_result)
        search_result = await _call_tool(
            session, "file_search", {"pattern": "Eidosian", "root_path": str(SANDBOX)}
        )
        self.assertIn(str(sandbox_file), search_result)
        await _call_tool(
            session,
            "file_write",
            {"file_path": str(sandbox_file_2), "content": "Eidosian MCP sandbox"},
        )
        dupes = await _call_tool(session, "file_find_duplicates", {"root_path": str(SANDBOX)})
        self.assertIn(str(sandbox_file), dupes)

        blocked = await _call_tool(
            session,
            "run_shell_command",
            {"command": "echo blocked > /tmp/should_not_write"},
        )
        self.assertIn("Unsafe command blocked", blocked)
        allowed = await _call_tool(
            session,
            "run_shell_command",
            {
                "command": f"echo guard_ok > {sandbox_file}",
                "transaction_paths": [str(sandbox_file)],
                "safe_mode": True,
            },
        )
        self.assertIn("\"exit_code\": 0", allowed)

        delete_result = await _call_tool(session, "file_delete", {"file_path": str(sandbox_file)})
        self.assertIn("Committed file_delete", delete_result)
        restore_result = await _call_tool(session, "file_restore", {"file_path": str(sandbox_file)})
        self.assertIn("Restored", restore_result)

        txn_list = await _call_tool(session, "transaction_list", {"limit": 3})
        self.assertIn("timestamp", txn_list)

        gis_snapshot = await _call_tool(session, "gis_snapshot")
        self.assertIn("Snapshot created", gis_snapshot)
        gis_set = await _call_tool(session, "gis_set", {"key": "mcp.test.key", "value": "ok"})
        self.assertIn("GIS updated", gis_set)
        gis_get = await _call_tool(session, "gis_get", {"key": "mcp.test.key"})
        self.assertIn("ok", gis_get)
        gis_txn = _extract_txn_id(gis_snapshot)
        gis_restore = await _call_tool(
            session, "gis_restore", {"transaction_id": gis_txn} if gis_txn else {}
        )
        self.assertIn("GIS restored", gis_restore)

        type_snapshot = await _call_tool(session, "type_snapshot")
        self.assertIn("Snapshot created", type_snapshot)
        type_register = await _call_tool(
            session,
            "type_register",
            {"name": "mcp_test", "schema": {"type": "object", "properties": {"x": {"type": "string"}}}},
        )
        self.assertTrue(
            type_register.startswith("Updated") or type_register == "No-op: Schema unchanged"
        )
        type_validate = await _call_tool(
            session, "type_validate", {"name": "mcp_test", "data": {"x": "ok"}}
        )
        self.assertEqual("valid", type_validate)
        type_txn = _extract_txn_id(type_snapshot)
        type_restore = await _call_tool(
            session,
            "type_restore_snapshot",
            {"transaction_id": type_txn} if type_txn else {},
        )
        self.assertIn("Type schemas restored", type_restore)

        mem_snapshot = await _call_tool(session, "memory_snapshot")
        self.assertIn("Snapshot created", mem_snapshot)
        mem_add = await _call_tool(session, "memory_add", {"content": "MCP memory test", "is_fact": True})
        self.assertIn("Memory added", mem_add)
        mem_retrieve = await _call_tool(session, "memory_retrieve", {"query": "MCP memory", "limit": 2})
        self.assertIn("MCP memory test", mem_retrieve)
        mem_stats = await _call_tool(session, "memory_stats")
        self.assertIn("count", mem_stats)
        mem_txn = _extract_txn_id(mem_snapshot)
        mem_restore = await _call_tool(
            session, "memory_restore", {"transaction_id": mem_txn} if mem_txn else {}
        )
        self.assertIn("Memory restored", mem_restore)

        sem_snapshot = await _call_tool(session, "memory_snapshot_semantic")
        self.assertIn("Snapshot created", sem_snapshot)
        sem_add = await _call_tool(session, "memory_add_semantic", {"content": "Semantic MCP test"})
        self.assertIn("Stored", sem_add)
        sem_search = await _call_tool(session, "memory_search", {"query": "Semantic MCP"})
        self.assertIn("Semantic MCP test", sem_search)
        sem_clear = await _call_tool(session, "memory_clear_semantic")
        self.assertIn("Memory cleared", sem_clear)
        sem_txn = _extract_txn_id(sem_snapshot)
        sem_restore = await _call_tool(
            session, "memory_restore_semantic", {"transaction_id": sem_txn} if sem_txn else {}
        )
        self.assertIn("Semantic memory restored", sem_restore)

        kb_add_1 = await _call_tool(
            session, "kb_add", {"fact": "MCP test fact A", "tags": ["mcp", "test"]}
        )
        kb_add_2 = await _call_tool(
            session, "kb_add", {"fact": "MCP test fact B", "tags": ["mcp", "test"]}
        )
        self.assertIn("Added node", kb_add_1)
        node_a = _extract_node_id(kb_add_1)
        node_b = _extract_node_id(kb_add_2)
        kb_search = await _call_tool(session, "kb_search", {"query": "MCP test"})
        self.assertIn("MCP test fact A", kb_search)
        kb_tag = await _call_tool(session, "kb_get_by_tag", {"tag": "mcp"})
        self.assertIn("MCP test fact B", kb_tag)
        if node_a and node_b:
            kb_link = await _call_tool(
                session, "kb_link", {"node_id_a": node_a, "node_id_b": node_b}
            )
            self.assertIn("Linked", kb_link)
            kb_delete = await _call_tool(session, "kb_delete", {"node_id": node_a})
            self.assertIn("Deleted node", kb_delete)
            kb_restore = await _call_tool(session, "kb_restore", {})
            self.assertIn("Knowledge base restored", kb_restore)

        grag_local = await _call_tool(session, "grag_query_local", {"query": "MCP test"})
        self.assertIn("Simulated", grag_local)
        grag_global = await _call_tool(session, "grag_query", {"query": "MCP test"})
        self.assertIn("Simulated", grag_global)
        grag_index = await _call_tool(session, "grag_index", {"scan_roots": [str(SANDBOX)]})
        self.assertIn("Simulated", grag_index)

        refactor = await _call_tool(
            session,
            "refactor_analyze",
            {"path": "/home/lloyd/eidosian_forge/eidos_mcp/src/eidos_mcp/forge_loader.py"},
        )
        self.assertIn("file_info", refactor)

        agent = await _call_tool(session, "agent_run_task", {"objective": "List available tools"})
        self.assertIn("objective", agent)

        audit_add = await _call_tool(
            session,
            "audit_add_todo",
            {"section": "MCP Validation", "task_text": "Sandbox audit test"},
        )
        self.assertTrue(
            audit_add.startswith("Added") or audit_add == "No-op: Task already exists"
        )
        audit_review = await _call_tool(
            session,
            "audit_mark_reviewed",
            {"path": str(SANDBOX), "agent_id": "mcp-test", "scope": "shallow"},
        )
        self.assertIn("Marked reviewed", audit_review)

        run_tests = await _call_tool(session, "run_tests", {"test_command": "echo MCP_TEST"})
        self.assertIn("MCP_TEST", run_tests)
        venv_run = await _call_tool(
            session,
            "venv_run",
            {"venv_path": "/home/lloyd/eidosian_forge/eidosian_venv", "command": "python -V"},
        )
        self.assertIn("Python 3.12", venv_run)

    @pytest.mark.skip(reason="Integration test requires dedicated MCP server infrastructure")
    async def test_tools_end_to_end_http_and_stdio(self) -> None:
        if SANDBOX.exists():
            shutil.rmtree(SANDBOX)
        SANDBOX.mkdir(parents=True, exist_ok=True)
        # HTTP/SSE path
        proc = _start_http_server()
        try:
            _wait_for_health(proc)
            async with streamable_http_client(f"http://{MCP_HOST}:{MCP_PORT}/streamable-http") as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    await self._run_tool_flow(session)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        # stdio path
        params = StdioServerParameters(
            command=VENV_PYTHON,
            args=["-u", "-c", "import eidos_mcp.eidos_mcp_server as s; s.main()"],
            env={
                **os.environ,
                "PYTHONPATH": "/home/lloyd/eidosian_forge/eidos_mcp/src:/home/lloyd/eidosian_forge",
                "EIDOS_FORGE_DIR": "/home/lloyd/eidosian_forge",
                "EIDOS_MCP_TRANSPORT": "stdio",
            },
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await self._run_tool_flow(session)
