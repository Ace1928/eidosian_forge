from __future__ import annotations

import asyncio
import importlib.machinery
import importlib.util
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB_ROOT = ROOT / "lib"
for extra in (LIB_ROOT, ROOT / "eidos_mcp" / "src"):
    value = str(extra)
    if value not in sys.path:
        sys.path.insert(0, value)

from eidosian_agent.local_mcp_agent import (
    build_tool_contracts,
    load_profile,
    normalize_profile,
    validate_tool_arguments,
)

SCRIPT_PATH = ROOT / "scripts" / "eidos_local_agent.py"


class _Tool:
    def __init__(self, name: str, description: str, input_schema: dict):
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class _ListTools:
    def __init__(self, tools: list[_Tool]):
        self.tools = tools


class _FakeSession:
    def __init__(self, tools: list[_Tool], responses: dict[str, str]):
        self._tools = tools
        self._responses = responses

    async def list_tools(self):
        return _ListTools(self._tools)

    async def call_tool(self, name, arguments=None):
        class _Text:
            type = "text"

            def __init__(self, text: str):
                self.text = text

        class _Result:
            structuredContent = None

            def __init__(self, text: str):
                self.content = [_Text(text)]

        return _Result(self._responses.get(name, "ok"))

    async def list_resources(self):
        class _Resource:
            def __init__(self, uri: str, name: str, description: str):
                self.uri = uri
                self.name = name
                self.description = description

        class _Result:
            def __init__(self):
                self.resources = [_Resource("memory://status", "memory-status", "memory status view")]

        return _Result()

    async def read_resource(self, uri):
        class _Item:
            def __init__(self, text: str):
                self.text = text
                self.blob = None

        class _Result:
            def __init__(self, text: str):
                self.contents = [_Item(text)]

        return _Result(f"resource:{uri}")


class _FakeCoordinator:
    def __init__(self):
        self.cleared = False
        self.heartbeats = []

    def can_allocate(self, **kwargs):
        return {"allowed": True, "reason": "ok"}

    def heartbeat(self, **kwargs):
        self.heartbeats.append(kwargs)
        return kwargs

    def clear_owner(self, owner, metadata=None):
        self.cleared = True
        return {"owner": owner, "metadata": metadata or {}}


def _load_script_module():
    loader = importlib.machinery.SourceFileLoader("eidos_local_agent", str(SCRIPT_PATH))
    spec = importlib.util.spec_from_loader("eidos_local_agent", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


def test_contract_schema_restricts_to_allowed_keys() -> None:
    profile = normalize_profile(
        "observer",
        {
            "allowed_tools": {
                "kb_search": {
                    "allowed_keys": ["query"],
                    "string_max_lengths": {"query": 32},
                }
            }
        },
    )
    tools, missing = build_tool_contracts(
        [
            _Tool(
                "kb_search",
                "Search knowledge",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            )
        ],
        profile,
    )
    assert missing == []
    schema = tools[0]["function"]["parameters"]
    assert set(schema["properties"].keys()) == {"query"}
    assert schema["additionalProperties"] is False
    assert schema["properties"]["query"]["maxLength"] == 32


def test_validate_tool_arguments_rejects_extra_keys() -> None:
    profile = normalize_profile(
        "observer",
        {
            "allowed_tools": {
                "kb_search": {
                    "allowed_keys": ["query"],
                    "string_max_lengths": {"query": 32},
                }
            }
        },
    )
    try:
        validate_tool_arguments("kb_search", {"query": "ok", "limit": 5}, profile, {})
    except ValueError as exc:
        assert "unexpected_keys" in str(exc)
    else:
        raise AssertionError("expected validation failure")


def test_validate_tool_arguments_enforces_bounds() -> None:
    profile = normalize_profile(
        "observer",
        {
            "allowed_tools": {
                "wf_search_terms": {
                    "allowed_keys": ["pattern", "limit"],
                    "integer_bounds": {"limit": {"min": 1, "max": 3}},
                }
            }
        },
    )
    try:
        validate_tool_arguments("wf_search_terms", {"pattern": "abc", "limit": 4}, profile, {})
    except ValueError as exc:
        assert "integer_out_of_bounds" in str(exc)
    else:
        raise AssertionError("expected validation failure")


def test_validate_tool_arguments_enforces_allowed_patterns() -> None:
    profile = normalize_profile(
        "observer",
        {
            "allowed_tools": {
                "run_shell_command": {
                    "allowed_keys": ["command"],
                    "allowed_patterns": {"command": [r"^(ls|find)(\s|$)"]},
                }
            }
        },
    )
    ok = validate_tool_arguments("run_shell_command", {"command": "ls -la"}, profile, {})
    assert ok["command"] == "ls -la"
    try:
        validate_tool_arguments("run_shell_command", {"command": "rm -rf tmp"}, profile, {})
    except ValueError as exc:
        assert "allowed_pattern_mismatch" in str(exc)
    else:
        raise AssertionError("expected validation failure")


def test_cli_wrapper_loads() -> None:
    module = _load_script_module()
    assert hasattr(module, "main")


def test_load_real_profile_contract() -> None:
    profile = load_profile(ROOT / "cfg" / "local_agent_profiles.json", "observer")
    assert profile.name == "observer"
    assert "kb_search" in profile.allowed_tools
    assert "read_resource" in profile.allowed_tools
    assert profile.max_mutating_calls == 0


def test_build_tool_contracts_includes_synthetic_resource_reader() -> None:
    profile = normalize_profile(
        "observer",
        {
            "allowed_tools": {
                "read_resource": {
                    "synthetic": "resource_read",
                    "allowed_keys": ["uri"],
                    "string_max_lengths": {"uri": 120},
                }
            }
        },
    )
    tools, missing = build_tool_contracts([], profile)
    assert missing == []
    assert tools[0]["function"]["name"] == "read_resource"


def test_agent_blank_urls_fall_back_to_defaults() -> None:
    from eidosian_agent.local_mcp_agent import DEFAULT_MCP_URL, DEFAULT_MODEL_URL, LocalMcpAgent

    agent = LocalMcpAgent(model_url="", mcp_url="", profile=load_profile(ROOT / "cfg" / "local_agent_profiles.json"))
    assert agent.model_url == DEFAULT_MODEL_URL
    assert agent.mcp_url == DEFAULT_MCP_URL


def test_stale_local_agent_owner_can_be_recovered(tmp_path: Path) -> None:
    from eidosian_agent.local_mcp_agent import LocalMcpAgent
    from eidosian_runtime import ForgeRuntimeCoordinator

    coordinator = ForgeRuntimeCoordinator(tmp_path / "forge_coordinator_status.json")
    coordinator.heartbeat(
        owner="local_mcp_agent:observer",
        task="local_agent:observer",
        state="running",
        active_models=[{"family": "ollama", "model": "qwen3.5:2b", "role": "local_agent:observer"}],
        metadata={"mode": "local_agent_cycle"},
    )
    payload = json.loads((tmp_path / "forge_coordinator_status.json").read_text(encoding="utf-8"))
    payload["updated_at"] = "2026-03-07T00:00:00+00:00"
    (tmp_path / "forge_coordinator_status.json").write_text(json.dumps(payload), encoding="utf-8")

    agent = LocalMcpAgent(
        coordinator=coordinator,
        profile=load_profile(ROOT / "cfg" / "local_agent_profiles.json"),
    )
    assert agent._recover_stale_own_lease(owner="local_mcp_agent:observer", stale_after_sec=30.0) is True


def test_run_cycle_returns_timeout_artifact(monkeypatch, tmp_path: Path) -> None:
    import eidosian_agent.local_mcp_agent as mod
    from eidosian_agent.local_mcp_agent import LocalMcpAgent
    from eidosian_runtime import ForgeRuntimeCoordinator

    @asynccontextmanager
    async def _fake_session_ctx(_root, url=None):
        yield _FakeSession([_Tool("diagnostics_ping", "Ping", {"type": "object", "properties": {}})], {}), "stdio"

    async def _fake_request_step(*args, **kwargs):
        raise TimeoutError("simulated timeout")

    monkeypatch.setattr(mod, "open_mcp_session", _fake_session_ctx)

    coordinator = ForgeRuntimeCoordinator(tmp_path / "forge_coordinator_status.json")
    agent = LocalMcpAgent(
        coordinator=coordinator,
        profile=load_profile(ROOT / "cfg" / "local_agent_profiles.json"),
        runtime_dir=tmp_path / "runtime",
    )
    monkeypatch.setattr(agent, "_request_step", _fake_request_step)
    result = asyncio.run(agent.run_cycle("health check", timeout_sec=1.0))
    assert result["status"] == "timeout"


def test_run_cycle_success_records_transport_and_resources(monkeypatch, tmp_path: Path) -> None:
    import eidosian_agent.local_mcp_agent as mod
    from eidosian_agent.local_mcp_agent import LocalMcpAgent
    from eidosian_runtime import ForgeRuntimeCoordinator

    @asynccontextmanager
    async def _fake_session_ctx(_root, url=None):
        yield _FakeSession([_Tool("diagnostics_ping", "Ping", {"type": "object", "properties": {}})], {}), "stdio"

    async def _fake_request_step(*args, **kwargs):
        return {"message": {"content": "done"}, "effective_thinking_mode": "on"}

    monkeypatch.setattr(mod, "open_mcp_session", _fake_session_ctx)

    coordinator = ForgeRuntimeCoordinator(tmp_path / "forge_coordinator_status.json")
    agent = LocalMcpAgent(
        coordinator=coordinator,
        profile=load_profile(ROOT / "cfg" / "local_agent_profiles.json"),
        runtime_dir=tmp_path / "runtime",
    )
    monkeypatch.setattr(agent, "_request_step", _fake_request_step)
    result = asyncio.run(agent.run_cycle("health check", timeout_sec=1.0))
    assert result["status"] == "success"
    assert result["mcp_transport"] == "stdio"
    assert result["resource_count"] == 1
    assert result["tool_contract_count"] >= 1


def test_run_cycle_can_read_allowed_resource(monkeypatch, tmp_path: Path) -> None:
    import eidosian_agent.local_mcp_agent as mod
    from eidosian_agent.local_mcp_agent import LocalMcpAgent
    from eidosian_runtime import ForgeRuntimeCoordinator

    @asynccontextmanager
    async def _fake_session_ctx(_root, url=None):
        yield _FakeSession([], {}), "stdio"

    calls = {"count": 0}

    async def _fake_request_step(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "read_resource", "arguments": {"uri": "memory://status"}}}
                    ]
                },
                "effective_thinking_mode": "on",
            }
        return {"message": {"content": "done"}, "effective_thinking_mode": "on"}

    monkeypatch.setattr(mod, "open_mcp_session", _fake_session_ctx)

    coordinator = ForgeRuntimeCoordinator(tmp_path / "forge_coordinator_status.json")
    agent = LocalMcpAgent(
        coordinator=coordinator,
        profile=load_profile(ROOT / "cfg" / "local_agent_profiles.json"),
        runtime_dir=tmp_path / "runtime",
    )
    monkeypatch.setattr(agent, "_request_step", _fake_request_step)
    result = asyncio.run(agent.run_cycle("read resource", timeout_sec=1.0))
    assert result["status"] == "success"
    assert result["tool_calls"] == 1
    assert result["cycle_log"][0]["tool"] == "read_resource"
