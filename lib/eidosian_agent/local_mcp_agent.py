from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from eidosian_core.ports import get_service_url
from eidosian_runtime import ForgeRuntimeCoordinator
from mcp import ClientSession, StdioServerParameters
from mcp.client.session import ClientSession as SessionType
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

FORGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = FORGE_ROOT / "cfg" / "local_agent_profiles.json"
DEFAULT_RUNTIME_DIR = FORGE_ROOT / "data" / "runtime" / "local_mcp_agent"
DEFAULT_MCP_URL = os.environ.get(
    "EIDOS_MCP_URL",
    get_service_url("eidos_mcp", default_port=8928, default_host="127.0.0.1", default_path="/mcp"),
)
DEFAULT_MODEL_URL = get_service_url(
    "ollama_qwen_http", default_port=8938, default_host="127.0.0.1", default_path=""
).rstrip("/")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _exception_chain(exc: BaseException) -> list[str]:
    messages: list[str] = []
    if isinstance(exc, BaseExceptionGroup):
        messages.append(type(exc).__name__)
        for child in exc.exceptions:
            messages.extend(_exception_chain(child))
        return messages
    text = str(exc).strip()
    if text:
        messages.append(f"{type(exc).__name__}: {text}")
    else:
        messages.append(type(exc).__name__)
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, BaseException):
        messages.extend(_exception_chain(cause))
    return messages


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


@dataclass(frozen=True)
class AgentProfile:
    name: str
    description: str
    max_steps: int
    max_tool_calls: int
    max_mutating_calls: int
    thinking_mode: str
    temperature: float
    max_tokens: int
    keep_alive: str
    max_observation_chars: int
    system_prompt: str
    allowed_tools: dict[str, dict[str, Any]]


class ToolPolicyError(ValueError):
    pass


class ModelResponseError(RuntimeError):
    pass


MUTATING_TOOL_NAMES = {
    "kb_add",
    "tiered_remember",
    "wf_add_term",
    "wf_add_relationship",
}


def _normalize_tool_rule(name: str, payload: Any) -> dict[str, Any]:
    rule = _as_dict(payload)
    allowed_keys = [str(x) for x in _as_list(rule.get("allowed_keys"))]
    string_max_lengths = {str(k): int(v) for k, v in _as_dict(rule.get("string_max_lengths")).items() if str(k).strip()}
    integer_bounds = {}
    for key, bounds in _as_dict(rule.get("integer_bounds")).items():
        bounds_dict = _as_dict(bounds)
        integer_bounds[str(key)] = {
            "min": int(bounds_dict.get("min", -(2**31))),
            "max": int(bounds_dict.get("max", 2**31 - 1)),
        }
    allowed_values = {
        str(k): [str(v) for v in _as_list(vals)]
        for k, vals in _as_dict(rule.get("allowed_values")).items()
        if str(k).strip()
    }
    const_arguments = {str(k): v for k, v in _as_dict(rule.get("const_arguments")).items()}
    path_prefixes = {
        str(k): [str(v) for v in _as_list(vals)]
        for k, vals in _as_dict(rule.get("path_prefixes")).items()
        if str(k).strip()
    }
    blocked_patterns = {
        str(k): [str(v) for v in _as_list(vals)]
        for k, vals in _as_dict(rule.get("blocked_patterns")).items()
        if str(k).strip()
    }
    allowed_patterns = {
        str(k): [str(v) for v in _as_list(vals)]
        for k, vals in _as_dict(rule.get("allowed_patterns")).items()
        if str(k).strip()
    }
    return {
        "synthetic": str(rule.get("synthetic") or "").strip().lower(),
        "mode": str(rule.get("mode") or ("mutating" if name in MUTATING_TOOL_NAMES else "read")).strip().lower(),
        "allowed_keys": allowed_keys,
        "string_max_lengths": string_max_lengths,
        "integer_bounds": integer_bounds,
        "allowed_values": allowed_values,
        "const_arguments": const_arguments,
        "path_prefixes": path_prefixes,
        "blocked_patterns": blocked_patterns,
        "allowed_patterns": allowed_patterns,
        "max_calls_per_cycle": max(1, int(rule.get("max_calls_per_cycle", 1) or 1)),
    }


def normalize_profile(name: str, payload: Any) -> AgentProfile:
    raw = _as_dict(payload)
    allowed_tools = {
        str(tool_name): _normalize_tool_rule(str(tool_name), rule)
        for tool_name, rule in _as_dict(raw.get("allowed_tools")).items()
        if str(tool_name).strip()
    }
    return AgentProfile(
        name=name,
        description=str(raw.get("description") or "").strip(),
        max_steps=max(1, int(raw.get("max_steps", 6) or 6)),
        max_tool_calls=max(1, int(raw.get("max_tool_calls", 8) or 8)),
        max_mutating_calls=max(0, int(raw.get("max_mutating_calls", 0) or 0)),
        thinking_mode=str(raw.get("thinking_mode") or "on").strip().lower(),
        temperature=float(raw.get("temperature", 0.2) or 0.2),
        max_tokens=max(128, int(raw.get("max_tokens", 2048) or 2048)),
        keep_alive=str(raw.get("keep_alive") or "2h").strip(),
        max_observation_chars=max(256, int(raw.get("max_observation_chars", 2400) or 2400)),
        system_prompt=str(raw.get("system_prompt") or "").strip(),
        allowed_tools=allowed_tools,
    )


def load_profile(path: str | Path | None = None, name: str = "observer") -> AgentProfile:
    policy_path = Path(path or DEFAULT_POLICY_PATH)
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    profiles = _as_dict(payload.get("profiles"))
    if name not in profiles:
        raise KeyError(f"Profile not found: {name}")
    return normalize_profile(name, profiles[name])


def contract_schema(schema: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    original = _as_dict(schema)
    properties = _as_dict(original.get("properties"))
    required = [str(x) for x in _as_list(original.get("required"))]
    allowed_keys = list(rule.get("allowed_keys") or properties.keys())
    contracted_properties: dict[str, Any] = {}
    for key in allowed_keys:
        prop = dict(_as_dict(properties.get(key)))
        if key in _as_dict(rule.get("string_max_lengths")):
            prop["maxLength"] = int(rule["string_max_lengths"][key])
        if key in _as_dict(rule.get("integer_bounds")):
            bounds = _as_dict(rule["integer_bounds"][key])
            prop["minimum"] = int(bounds.get("min", -(2**31)))
            prop["maximum"] = int(bounds.get("max", 2**31 - 1))
        if key in _as_dict(rule.get("allowed_values")):
            prop["enum"] = list(rule["allowed_values"][key])
        if key in _as_dict(rule.get("const_arguments")):
            prop["const"] = rule["const_arguments"][key]
        if not prop:
            prop = {"type": "string"}
        contracted_properties[key] = prop
    return {
        "type": "object",
        "properties": contracted_properties,
        "required": [key for key in required if key in contracted_properties],
        "additionalProperties": False,
    }


def build_tool_contracts(discovered_tools: list[Any], profile: AgentProfile) -> tuple[list[dict[str, Any]], list[str]]:
    by_name = {str(getattr(tool, "name", "")): tool for tool in discovered_tools}
    tools: list[dict[str, Any]] = []
    missing: list[str] = []
    for name, rule in profile.allowed_tools.items():
        if str(rule.get("synthetic") or "") == "resource_read":
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": "Read an allowed MCP resource by URI under the local-agent policy contract.",
                        "parameters": contract_schema(
                            {
                                "type": "object",
                                "properties": {"uri": {"type": "string"}},
                                "required": ["uri"],
                            },
                            rule,
                        ),
                    },
                }
            )
            continue
        tool = by_name.get(name)
        if tool is None:
            missing.append(name)
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(getattr(tool, "description", "") or "").strip(),
                    "parameters": contract_schema(_as_dict(getattr(tool, "inputSchema", {})), rule),
                },
            }
        )
    return tools, sorted(missing)


def validate_tool_arguments(name: str, args: Any, profile: AgentProfile, call_counts: dict[str, int]) -> dict[str, Any]:
    if name not in profile.allowed_tools:
        raise ToolPolicyError(f"tool_not_allowed:{name}")
    if not isinstance(args, dict):
        raise ToolPolicyError(f"tool_arguments_not_object:{name}")
    rule = profile.allowed_tools[name]
    call_counts[name] = int(call_counts.get(name, 0)) + 1
    if call_counts[name] > int(rule.get("max_calls_per_cycle", 1)):
        raise ToolPolicyError(f"tool_call_budget_exceeded:{name}")
    allowed_keys = set(rule.get("allowed_keys") or [])
    extra_keys = sorted(str(key) for key in args.keys() if allowed_keys and str(key) not in allowed_keys)
    if extra_keys:
        raise ToolPolicyError(f"unexpected_keys:{name}:{','.join(extra_keys)}")
    cleaned: dict[str, Any] = {}
    for key, value in args.items():
        str_key = str(key)
        if str_key in _as_dict(rule.get("const_arguments")) and value != rule["const_arguments"][str_key]:
            raise ToolPolicyError(f"const_argument_mismatch:{name}:{str_key}")
        if isinstance(value, str):
            limit = _as_dict(rule.get("string_max_lengths")).get(str_key)
            if limit is not None and len(value) > int(limit):
                raise ToolPolicyError(f"string_too_long:{name}:{str_key}")
            allowed_values = _as_dict(rule.get("allowed_values")).get(str_key)
            if allowed_values and value not in allowed_values:
                raise ToolPolicyError(f"unexpected_value:{name}:{str_key}")
            allowed_patterns = _as_dict(rule.get("allowed_patterns")).get(str_key, [])
            if allowed_patterns and not any(re.search(pattern, value) for pattern in allowed_patterns):
                raise ToolPolicyError(f"allowed_pattern_mismatch:{name}:{str_key}")
            for pattern in _as_dict(rule.get("blocked_patterns")).get(str_key, []):
                if re.search(pattern, value):
                    raise ToolPolicyError(f"blocked_pattern:{name}:{str_key}")
            prefixes = _as_dict(rule.get("path_prefixes")).get(str_key, [])
            if prefixes and not any(value.startswith(prefix) for prefix in prefixes):
                raise ToolPolicyError(f"path_prefix_denied:{name}:{str_key}")
            cleaned[str_key] = value
            continue
        if isinstance(value, bool):
            cleaned[str_key] = value
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            bounds = _as_dict(rule.get("integer_bounds")).get(str_key)
            if bounds:
                lo = int(bounds.get("min", -(2**31)))
                hi = int(bounds.get("max", 2**31 - 1))
                if value < lo or value > hi:
                    raise ToolPolicyError(f"integer_out_of_bounds:{name}:{str_key}")
            cleaned[str_key] = value
            continue
        if isinstance(value, float):
            cleaned[str_key] = value
            continue
        if isinstance(value, list):
            cleaned[str_key] = value
            continue
        if isinstance(value, dict):
            cleaned[str_key] = value
            continue
        cleaned[str_key] = value
    for key, value in _as_dict(rule.get("const_arguments")).items():
        cleaned.setdefault(str(key), value)
    return cleaned


async def _call_tool(
    session: SessionType, name: str, arguments: dict[str, Any] | None = None, timeout_sec: float = 60.0
) -> str:
    result = await asyncio.wait_for(session.call_tool(name, arguments=arguments or {}), timeout=timeout_sec)
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
    return "\n".join([line for line in lines if line]).strip()


async def _http_ready(url: str) -> bool:
    base_url = url[:-4] if url.endswith("/mcp") else url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{base_url}/health")
        return response.status_code == 200
    except Exception:
        return False


async def _list_resources(session: SessionType, timeout_sec: float = 15.0) -> list[dict[str, str]]:
    if not hasattr(session, "list_resources"):
        return []
    try:
        result = await asyncio.wait_for(session.list_resources(), timeout=timeout_sec)
    except Exception:
        return []
    rows: list[dict[str, str]] = []
    for item in getattr(result, "resources", []) or []:
        rows.append(
            {
                "uri": str(getattr(item, "uri", "") or "").strip(),
                "name": str(getattr(item, "name", "") or "").strip(),
                "description": str(getattr(item, "description", "") or "").strip(),
            }
        )
    return rows


async def _read_resource(session: SessionType, uri: str, timeout_sec: float = 60.0) -> str:
    result = await asyncio.wait_for(session.read_resource(uri), timeout=timeout_sec)
    lines: list[str] = []
    for block in getattr(result, "contents", []) or []:
        text = getattr(block, "text", None)
        if text:
            lines.append(str(text))
            continue
        blob = getattr(block, "blob", None)
        if blob:
            lines.append(str(blob))
    return "\n".join([line for line in lines if line]).strip()


@asynccontextmanager
async def open_mcp_session(root: Path, url: str | None = None) -> AsyncIterator[tuple[ClientSession, str]]:
    target_url = str(url or DEFAULT_MCP_URL)
    if await _http_ready(target_url):
        async with streamable_http_client(target_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session, "streamable_http"
        return

    venv_python = root / "eidosian_venv" / "bin" / "python3"
    python_bin = str(venv_python if venv_python.exists() else Path(os.environ.get("PYTHON", "python3")))
    pythonpath = ":".join(
        [
            str(root / "lib"),
            str(root / "eidos_mcp" / "src"),
            str(root / "memory_forge" / "src"),
            str(root / "knowledge_forge" / "src"),
            str(root / "code_forge" / "src"),
            str(root / "ollama_forge" / "src"),
            str(root),
        ]
    )
    params = StdioServerParameters(
        command=python_bin,
        args=["-u", "-c", "import eidos_mcp.eidos_mcp_server as s; s.main()"],
        env={
            **os.environ,
            "PYTHONPATH": pythonpath,
            "EIDOS_FORGE_DIR": str(root),
            "EIDOS_MCP_TRANSPORT": "stdio",
            "EIDOS_MCP_STATELESS_HTTP": "1",
        },
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session, "stdio"


class LocalMcpAgent:
    def __init__(
        self,
        *,
        forge_root: str | Path | None = None,
        model: str = "qwen3.5:2b",
        model_url: str = DEFAULT_MODEL_URL,
        mcp_url: str = DEFAULT_MCP_URL,
        profile: AgentProfile | None = None,
        coordinator: ForgeRuntimeCoordinator | None = None,
        runtime_dir: str | Path | None = None,
    ) -> None:
        self.forge_root = Path(forge_root or FORGE_ROOT).resolve()
        self.model = str(model)
        self.model_url = str(model_url or DEFAULT_MODEL_URL).rstrip("/")
        self.mcp_url = str(mcp_url or DEFAULT_MCP_URL)
        self.profile = profile or load_profile(DEFAULT_POLICY_PATH, "observer")
        self.coordinator = coordinator or ForgeRuntimeCoordinator()
        self.runtime_dir = Path(runtime_dir or DEFAULT_RUNTIME_DIR)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.runtime_dir / "status.json"
        self.history_path = self.runtime_dir / "history.jsonl"

    def _owner(self) -> str:
        return f"local_mcp_agent:{self.profile.name}"

    def _lease_models(self) -> list[dict[str, Any]]:
        return [{"family": "ollama", "model": self.model, "role": f"local_agent:{self.profile.name}"}]

    def _lease_metadata(self, objective: str) -> dict[str, Any]:
        return {
            "exclusive": False,
            "mode": "local_agent_cycle",
            "profile": self.profile.name,
            "objective": objective,
            "pid": os.getpid(),
            "host": socket.gethostname(),
        }

    def _recover_stale_own_lease(self, *, owner: str, stale_after_sec: float) -> bool:
        payload = self.coordinator.read()
        if str(payload.get("owner") or "") != owner:
            return False
        metadata = _as_dict(payload.get("metadata"))
        if str(metadata.get("mode") or "") != "local_agent_cycle":
            return False
        updated_at = _parse_iso_utc(payload.get("updated_at"))
        if updated_at is None:
            return False
        age = (datetime.now(timezone.utc) - updated_at).total_seconds()
        if age < max(30.0, float(stale_after_sec)):
            return False
        self.coordinator.clear_owner(
            owner,
            metadata={
                "exclusive": False,
                "mode": "local_agent_cycle",
                "profile": self.profile.name,
                "released_reason": "stale_owner_recovery",
            },
        )
        return True

    def _system_prompt(self) -> str:
        return self.profile.system_prompt.strip() or (
            "You are a guarded local Eidosian agent. Use tools only when needed. "
            "You must stay inside the exposed tool contract and stop when you have enough evidence to answer."
        )

    def _write_status(self, payload: dict[str, Any]) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _append_history(self, payload: dict[str, Any]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    async def _chat_payload(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        thinking_mode: str,
        timeout_sec: float,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {
                "temperature": self.profile.temperature,
                "num_predict": self.profile.max_tokens,
            },
            "think": thinking_mode == "on",
            "keep_alive": self.profile.keep_alive,
        }
        async with httpx.AsyncClient(timeout=max(10.0, float(timeout_sec))) as client:
            response = await client.post(f"{self.model_url}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()

    async def _request_step(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_sec: float,
    ) -> dict[str, Any]:
        preferred = self.profile.thinking_mode
        attempted = [preferred]
        if preferred == "on":
            attempted.append("off")
        last: dict[str, Any] = {}
        for mode in attempted:
            last = await self._chat_payload(messages=messages, tools=tools, thinking_mode=mode, timeout_sec=timeout_sec)
            message = _as_dict(last.get("message"))
            content = str(message.get("content") or "").strip()
            tool_calls = _as_list(message.get("tool_calls"))
            if content or tool_calls:
                last["effective_thinking_mode"] = mode
                return last
        last["effective_thinking_mode"] = attempted[-1]
        return last

    async def run_cycle(self, objective: str, *, timeout_sec: float = 1800.0) -> dict[str, Any]:
        profile = self.profile
        owner = self._owner()
        call_counts: dict[str, int] = {}
        mutating_calls = 0
        total_tool_calls = 0
        total_tool_latency_ms = 0
        cycle_log: list[dict[str, Any]] = []
        final_message = ""
        effective_thinking_mode = profile.thinking_mode
        mcp_transport = ""
        list_tools_ms = 0
        resource_count = 0
        resource_sample: list[dict[str, str]] = []
        tool_contract_count = 0
        allocation = self.coordinator.can_allocate(
            owner=owner, requested_models=self._lease_models(), allow_same_owner=False
        )
        if (
            not allocation.get("allowed")
            and str(allocation.get("reason") or "") in {"instance_budget_exceeded", "family_budget_exceeded"}
            and self._recover_stale_own_lease(owner=owner, stale_after_sec=max(90.0, float(timeout_sec) + 15.0))
        ):
            allocation = self.coordinator.can_allocate(
                owner=owner,
                requested_models=self._lease_models(),
                allow_same_owner=False,
            )
        if not allocation.get("allowed"):
            result = {
                "contract": "eidos.local_mcp_agent.result.v1",
                "status": "blocked",
                "blocked_reason": allocation.get("reason"),
                "objective": objective,
                "profile": profile.name,
                "owner": owner,
                "created_at": _now_utc(),
            }
            self._write_status(result)
            self._append_history(result)
            return result

        self.coordinator.heartbeat(
            owner=owner,
            task=f"local_agent:{profile.name}",
            state="running",
            active_models=self._lease_models(),
            metadata=self._lease_metadata(objective),
        )

        try:
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": (
                        f"Objective: {objective}\n"
                        f"Profile: {profile.name}\n"
                        f"Max steps: {profile.max_steps}\n"
                        "Use tools only when needed. If enough evidence is available, answer directly."
                    ),
                },
            ]
            async with open_mcp_session(self.forge_root, url=self.mcp_url) as opened:
                session, mcp_transport = opened
                list_started = time.perf_counter()
                discovered = await asyncio.wait_for(session.list_tools(), timeout=30.0)
                list_tools_ms = int(round((time.perf_counter() - list_started) * 1000.0))
                resources = await _list_resources(session)
                resource_count = len(resources)
                resource_sample = resources[:6]
                tool_contracts, missing_tools = build_tool_contracts(list(discovered.tools), profile)
                tool_contract_count = len(tool_contracts)
                if resource_sample:
                    resource_lines = []
                    for resource in resource_sample[:4]:
                        label = resource.get("name") or resource.get("uri") or "resource"
                        desc = resource.get("description") or ""
                        resource_lines.append(f"- {label}: {desc}".strip())
                    messages.append(
                        {
                            "role": "system",
                            "content": "Available MCP resources:\n" + "\n".join(resource_lines),
                        }
                    )
                for step_index in range(profile.max_steps):
                    if total_tool_calls >= profile.max_tool_calls:
                        break
                    raw_response = await self._request_step(
                        messages=messages, tools=tool_contracts, timeout_sec=timeout_sec
                    )
                    effective_thinking_mode = str(
                        raw_response.get("effective_thinking_mode") or effective_thinking_mode
                    )
                    message = _as_dict(raw_response.get("message"))
                    tool_calls = _as_list(message.get("tool_calls"))
                    content = str(message.get("content") or "").strip()
                    if content:
                        final_message = content
                    assistant_message: dict[str, Any] = {"role": "assistant"}
                    if content:
                        assistant_message["content"] = content
                    if tool_calls:
                        assistant_message["tool_calls"] = tool_calls
                    messages.append(assistant_message)
                    if not tool_calls:
                        break
                    for tool_call in tool_calls:
                        function = _as_dict(_as_dict(tool_call).get("function"))
                        name = str(function.get("name") or "").strip()
                        raw_args = function.get("arguments") if isinstance(function.get("arguments"), dict) else {}
                        validated_args = validate_tool_arguments(name, raw_args, profile, call_counts)
                        if profile.allowed_tools[name]["mode"] == "mutating":
                            mutating_calls += 1
                            if mutating_calls > profile.max_mutating_calls:
                                raise ToolPolicyError(f"mutating_call_budget_exceeded:{name}")
                        total_tool_calls += 1
                        tool_started = time.perf_counter()
                        if str(profile.allowed_tools[name].get("synthetic") or "") == "resource_read":
                            observation = await _read_resource(
                                session,
                                str(validated_args.get("uri") or ""),
                                timeout_sec=timeout_sec,
                            )
                        else:
                            observation = await _call_tool(session, name, validated_args, timeout_sec=timeout_sec)
                        duration_ms = int(round((time.perf_counter() - tool_started) * 1000.0))
                        total_tool_latency_ms += duration_ms
                        observation = observation[: profile.max_observation_chars]
                        messages.append({"role": "tool", "tool_name": name, "content": observation})
                        cycle_log.append(
                            {
                                "step": step_index + 1,
                                "tool": name,
                                "arguments": validated_args,
                                "duration_ms": duration_ms,
                                "observation_chars": len(observation),
                            }
                        )
                result = {
                    "contract": "eidos.local_mcp_agent.result.v1",
                    "status": "success",
                    "created_at": _now_utc(),
                    "objective": objective,
                    "profile": profile.name,
                    "thinking_mode": profile.thinking_mode,
                    "effective_thinking_mode": effective_thinking_mode,
                    "mcp_transport": mcp_transport,
                    "list_tools_ms": list_tools_ms,
                    "resource_count": resource_count,
                    "resource_sample": resource_sample,
                    "tool_contract_count": tool_contract_count,
                    "tool_calls": total_tool_calls,
                    "tool_latency_ms_total": total_tool_latency_ms,
                    "mutating_calls": mutating_calls,
                    "missing_tools": missing_tools,
                    "final_message": final_message,
                    "cycle_log": cycle_log,
                }
                self._write_status(result)
                self._append_history(result)
                return result
        except Exception as exc:
            error_chain = _exception_chain(exc)
            joined_error = " | ".join(error_chain)
            error_type = type(exc).__name__
            status = "timeout" if "timeout" in joined_error.lower() else "error"
            result = {
                "contract": "eidos.local_mcp_agent.result.v1",
                "status": status,
                "created_at": _now_utc(),
                "objective": objective,
                "profile": profile.name,
                "thinking_mode": profile.thinking_mode,
                "effective_thinking_mode": effective_thinking_mode,
                "mcp_transport": mcp_transport,
                "list_tools_ms": list_tools_ms,
                "resource_count": resource_count,
                "resource_sample": resource_sample,
                "tool_contract_count": tool_contract_count,
                "tool_calls": total_tool_calls,
                "tool_latency_ms_total": total_tool_latency_ms,
                "mutating_calls": mutating_calls,
                "final_message": final_message,
                "cycle_log": cycle_log,
                "error_type": error_type,
                "error": joined_error or str(exc),
                "error_chain": error_chain,
            }
            self._write_status(result)
            self._append_history(result)
            return result
        finally:
            self.coordinator.clear_owner(
                owner,
                metadata={
                    "exclusive": False,
                    "mode": "local_agent_cycle",
                    "profile": profile.name,
                    "task": f"local_agent:{profile.name}",
                    "released_reason": "cycle_finished",
                },
            )

    async def run_continuous(
        self,
        objective: str,
        *,
        interval_sec: float = 120.0,
        max_cycles: int = 0,
        timeout_sec: float = 1800.0,
    ) -> dict[str, Any]:
        count = 0
        latest: dict[str, Any] = {}
        while True:
            count += 1
            latest = await self.run_cycle(objective, timeout_sec=timeout_sec)
            if max_cycles > 0 and count >= max_cycles:
                return latest
            await asyncio.sleep(max(1.0, float(interval_sec)))


def cli_entry(
    *,
    objective: str,
    profile_name: str = "observer",
    policy_path: str | Path | None = None,
    model: str = "qwen3.5:2b",
    model_url: str = DEFAULT_MODEL_URL,
    mcp_url: str = DEFAULT_MCP_URL,
    continuous: bool = False,
    interval_sec: float = 120.0,
    max_cycles: int = 0,
    timeout_sec: float = 1800.0,
    keep_alive: str = "",
) -> dict[str, Any]:
    profile = load_profile(path=policy_path or DEFAULT_POLICY_PATH, name=profile_name)
    if keep_alive.strip():
        profile = AgentProfile(
            name=profile.name,
            description=profile.description,
            max_steps=profile.max_steps,
            max_tool_calls=profile.max_tool_calls,
            max_mutating_calls=profile.max_mutating_calls,
            thinking_mode=profile.thinking_mode,
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            keep_alive=keep_alive.strip(),
            max_observation_chars=profile.max_observation_chars,
            system_prompt=profile.system_prompt,
            allowed_tools=profile.allowed_tools,
        )
    agent = LocalMcpAgent(
        forge_root=FORGE_ROOT,
        model=model,
        model_url=model_url,
        mcp_url=mcp_url,
        profile=profile,
    )
    if continuous:
        return asyncio.run(
            agent.run_continuous(
                objective,
                interval_sec=interval_sec,
                max_cycles=max_cycles,
                timeout_sec=timeout_sec,
            )
        )
    return asyncio.run(agent.run_cycle(objective, timeout_sec=timeout_sec))
