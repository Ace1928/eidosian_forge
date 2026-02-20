#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class CallOutcome:
    name: str
    status: str
    message: str
    arguments: dict[str, Any] | None = None


def _extract_text(result: Any) -> str:
    lines: list[str] = []
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        try:
            lines.append(json.dumps(structured, ensure_ascii=False, default=str))
        except Exception:
            lines.append(str(structured))
    for block in getattr(result, "content", []) or []:
        if getattr(block, "type", None) == "text":
            lines.append(getattr(block, "text", ""))
    return "\n".join([line for line in lines if line])


def _default_value(
    key: str,
    prop_schema: dict[str, Any],
    context: dict[str, Any],
    state: dict[str, Any],
) -> Any:
    if key in state and state.get(key) is not None:
        return state[key]
    if key in context:
        return context[key]
    if "enum" in prop_schema and prop_schema["enum"]:
        return prop_schema["enum"][0]

    value_type = prop_schema.get("type")
    if value_type == "string":
        return f"{key}-value"
    if value_type == "integer":
        return 1
    if value_type == "number":
        return 0.5
    if value_type == "boolean":
        return False
    if value_type == "array":
        return []
    if value_type == "object":
        return {}
    return "value"


def _build_arguments(
    tool_name: str,
    input_schema: dict[str, Any],
    context: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    properties = input_schema.get("properties", {}) if isinstance(input_schema, dict) else {}
    required = input_schema.get("required", []) if isinstance(input_schema, dict) else []

    args: dict[str, Any] = {}
    for key in required:
        prop_schema = properties.get(key, {}) if isinstance(properties, dict) else {}
        args[key] = _default_value(key, prop_schema, context, state)

    # Tool-specific safe overrides.
    if tool_name == "run_shell_command":
        sandbox_file = context["sandbox_file_2"]
        return {
            "command": f"echo audit_ok > {sandbox_file}",
            "safe_mode": True,
            "transaction_paths": [sandbox_file],
            "idempotency_key": context["idempotency_key"],
        }
    if tool_name == "run_tests":
        return {"test_command": "python -V"}
    if tool_name == "venv_run":
        return {"venv_path": context["venv_path"], "command": "python -V", "timeout_sec": 30}
    if tool_name == "refactor_analyze":
        return {"path": context["refactor_target"]}
    if tool_name == "file_create":
        return {"file_path": context["sandbox_file_1"]}
    if tool_name == "file_write":
        return {"file_path": context["sandbox_file_1"], "content": "audit content", "overwrite": True}
    if tool_name == "file_read":
        return {"file_path": context["sandbox_file_1"]}
    if tool_name == "file_search":
        return {"pattern": "audit", "root_path": context["sandbox_dir"], "max_results": 10}
    if tool_name == "file_delete":
        return {"file_path": context["sandbox_file_1"]}
    if tool_name == "file_restore":
        return {"file_path": context["sandbox_file_1"]}
    if tool_name == "file_find_duplicates":
        return {"root_path": context["sandbox_dir"], "max_groups": 5}
    if tool_name == "kb_ingest_memory":
        return {"memory_path": context["memory_path"], "tags": ["audit"]}
    if tool_name in {"tika_extract_file", "tika_get_metadata", "tika_ingest_file"}:
        return {"file_path": context["sandbox_file_1"]}
    if tool_name == "tika_ingest_directory":
        return {"directory": context["sandbox_dir"], "recursive": True}
    if tool_name in {"tika_extract_url", "tika_ingest_url"}:
        return {"url": "https://example.com"}
    if tool_name == "consciousness_kernel_benchmark":
        return {"ticks": 2, "persist": False}
    if tool_name == "consciousness_kernel_trial":
        return {
            "kind": "noise",
            "target": "attention",
            "magnitude": 0.2,
            "duration_s": 1.0,
            "ticks": 1,
            "persist": False,
        }
    if tool_name == "consciousness_kernel_full_benchmark":
        return {
            "rounds": 1,
            "bench_ticks": 2,
            "trial_ticks": 1,
            "run_mcp": False,
            "run_llm": False,
            "persist": False,
        }
    if tool_name == "mcp_self_upgrade":
        # Intentionally not called: side-effecting self mutation and git operations.
        return {}

    return args


async def run_audit(root: Path, timeout_sec: float) -> dict[str, Any]:
    home = Path.home()
    venv_python = root / "eidosian_venv/bin/python3"
    python_bin = str(venv_python if venv_python.exists() else Path(sys.executable))
    pythonpath = f"{root}/eidos_mcp/src:{root}"

    sandbox_dir = home / ".eidosian" / "tmp" / "mcp_tool_audit"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    sandbox_file_1 = sandbox_dir / "sample.txt"
    sandbox_file_2 = sandbox_dir / "sample_copy.txt"
    sandbox_file_1.write_text("audit sample", encoding="utf-8")
    sandbox_file_2.write_text("audit sample", encoding="utf-8")

    context: dict[str, Any] = {
        "sandbox_dir": str(sandbox_dir),
        "sandbox_file_1": str(sandbox_file_1),
        "sandbox_file_2": str(sandbox_file_2),
        "venv_path": str(root / "eidosian_venv"),
        "refactor_target": str(root / "eidos_mcp/src/eidos_mcp/forge_loader.py"),
        "memory_path": str(root / "data/memory/memory_entries.json"),
        "idempotency_key": f"mcp-audit-{int(time.time())}",
        "query": "termux audit",
        "content": "termux audit content",
        "fact": f"termux audit fact {int(time.time())}",
        "tags": ["termux", "audit"],
        "section": "MCP Validation",
        "task_text": "Run exhaustive MCP tool audit",
        "agent_id": "mcp-audit",
        "objective": "List available tools",
        "target_tier": "working",
        "tier": "working",
        "namespace": "task",
        "importance": 0.5,
        "name": "mcp_audit_type",
        "schema": {"type": "object", "properties": {"x": {"type": "string"}}},
        "data": {"x": "ok"},
        "pattern": "audit",
        "term": "eidos",
        "term1": "eidos",
        "term2": "forge",
        "relation_type": "related_to",
        "text": "Eidosian Forge runs in Termux.",
        "sort": "new",
        "limit": 3,
        "submolt": "general",
        "title": "Termux Audit Post",
        "message": "Hello from MCP audit",
        "answer": "42",
        "verification_code": "test-code",
        "agent_name": "eidos",
        "default": "fallback",
        "max_results": 3,
        "include_memory": True,
        "include_knowledge": True,
        "include_self": True,
        "include_user": True,
        "include_task": True,
        "format": "json",
        "content_type": "text/plain",
    }

    # IDs discovered as tools run.
    state: dict[str, Any] = {
        "memory_id": None,
        "item_id": None,
        "node_id": None,
        "post_id": None,
        "comment_id": None,
        "conversation_id": None,
        "transaction_id": None,
    }

    params = StdioServerParameters(
        command=python_bin,
        args=["-u", "-c", "import eidos_mcp.eidos_mcp_server as s; s.main()"],
        env={
            **os.environ,
            "PYTHONPATH": pythonpath,
            "EIDOS_FORGE_DIR": str(root),
            "EIDOS_MCP_TRANSPORT": "stdio",
        },
    )

    resource_outcomes: list[CallOutcome] = []
    tool_outcomes: list[CallOutcome] = []

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            resources = await session.list_resources()
            for res in resources.resources:
                try:
                    fetched = await asyncio.wait_for(session.read_resource(res.uri), timeout=timeout_sec)
                    resource_outcomes.append(
                        CallOutcome(
                            name=res.uri,
                            status="ok",
                            message=f"contents={len(fetched.contents)}",
                        )
                    )
                except Exception as exc:
                    resource_outcomes.append(CallOutcome(name=res.uri, status="hard_fail", message=str(exc)))

            tools = await session.list_tools()
            for tool in sorted(tools.tools, key=lambda t: t.name):
                if tool.name == "mcp_self_upgrade":
                    tool_outcomes.append(
                        CallOutcome(
                            name=tool.name,
                            status="skipped",
                            message="Skipped intentionally: side-effecting self-upgrade operation.",
                        )
                    )
                    continue

                input_schema = getattr(tool, "inputSchema", {}) or {}
                arguments = _build_arguments(tool.name, input_schema, context, state)
                call_timeout = timeout_sec
                if tool.name in {"consciousness_kernel_full_benchmark"}:
                    # Full benchmark is intentionally heavier than single-probe tools.
                    call_timeout = max(timeout_sec, 30.0)
                try:
                    result = await asyncio.wait_for(
                        session.call_tool(tool.name, arguments=arguments),
                        timeout=call_timeout,
                    )
                    text = _extract_text(result)

                    if tool.name == "memory_add":
                        match = re.search(r"([0-9a-f]{8}-[0-9a-f\\-]{27})", text)
                        if match:
                            state["memory_id"] = match.group(1)
                            state["item_id"] = match.group(1)
                    elif tool.name == "kb_add":
                        match = re.search(r"Added node: ([0-9a-f\\-]+)", text)
                        if match:
                            state["node_id"] = match.group(1)
                    elif tool.name == "moltbook_create":
                        try:
                            payload = json.loads(text.split("\n")[0])
                            post_id = payload.get("post", {}).get("id") or payload.get("id")
                            if post_id:
                                state["post_id"] = post_id
                        except Exception:
                            pass

                    status = "soft_fail" if getattr(result, "isError", False) else "ok"
                    tool_outcomes.append(
                        CallOutcome(
                            name=tool.name,
                            status=status,
                            message=text[:400],
                            arguments=arguments,
                        )
                    )
                except Exception as exc:
                    tool_outcomes.append(
                        CallOutcome(
                            name=tool.name,
                            status="hard_fail",
                            message=str(exc),
                            arguments=arguments,
                        )
                    )

    hard_fail_tools = [o for o in tool_outcomes if o.status == "hard_fail"]
    soft_fail_tools = [o for o in tool_outcomes if o.status == "soft_fail"]
    skipped_tools = [o for o in tool_outcomes if o.status == "skipped"]
    hard_fail_resources = [o for o in resource_outcomes if o.status == "hard_fail"]

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "counts": {
            "tools_total": len(tool_outcomes),
            "resources_total": len(resource_outcomes),
            "tool_ok": len([o for o in tool_outcomes if o.status == "ok"]),
            "tool_soft_fail": len(soft_fail_tools),
            "tool_hard_fail": len(hard_fail_tools),
            "tool_skipped": len(skipped_tools),
            "resource_ok": len([o for o in resource_outcomes if o.status == "ok"]),
            "resource_hard_fail": len(hard_fail_resources),
        },
        "hard_fail_tools": [o.__dict__ for o in hard_fail_tools],
        "soft_fail_tools": [o.__dict__ for o in soft_fail_tools],
        "skipped_tools": [o.__dict__ for o in skipped_tools],
        "hard_fail_resources": [o.__dict__ for o in hard_fail_resources],
        "tool_outcomes": [o.__dict__ for o in tool_outcomes],
        "resource_outcomes": [o.__dict__ for o in resource_outcomes],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit all MCP tools/resources via stdio.")
    parser.add_argument("--timeout", type=float, default=20.0, help="Per-call timeout seconds.")
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory to write JSON report into.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    report = asyncio.run(run_audit(root=root, timeout_sec=args.timeout))

    report_dir = (root / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    report_path = report_dir / f"mcp_audit_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    counts = report["counts"]
    print(f"Report: {report_path}")
    print(
        "tools_total={tools_total} ok={tool_ok} soft_fail={tool_soft_fail} "
        "hard_fail={tool_hard_fail} skipped={tool_skipped}".format(**counts)
    )
    print("resources_total={resources_total} ok={resource_ok} hard_fail={resource_hard_fail}".format(**counts))

    return 1 if counts["tool_hard_fail"] or counts["resource_hard_fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
