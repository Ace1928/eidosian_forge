from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from eidosian_core import eidosian

from ..core import tool


FORGE_DIR = Path(os.environ.get("EIDOS_FORGE_DIR", "/home/lloyd/eidosian_forge"))
DEFAULT_BASE_URL = "https://www.moltbook.com/api/v1"
MOLTBOOK_TIMEOUT_SEC = float(os.environ.get("MOLTBOOK_TIMEOUT_SEC", "8.0"))


def _load_credentials() -> Dict[str, Optional[str]]:
    api_key = os.environ.get("MOLTBOOK_API_KEY")
    agent_name = os.environ.get("MOLTBOOK_AGENT_NAME")
    cred_path = os.environ.get(
        "MOLTBOOK_CREDENTIALS",
        str(Path.home() / ".config" / "moltbook" / "credentials.json"),
    )
    path = Path(cred_path).expanduser()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            api_key = api_key or data.get("api_key")
            agent_name = agent_name or data.get("agent_name")
        except Exception:
            pass
    return {"api_key": api_key, "agent_name": agent_name}


def _client() -> httpx.Client:
    base_url = os.environ.get("MOLTBOOK_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    creds = _load_credentials()
    api_key = creds.get("api_key") or ""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key
    timeout = httpx.Timeout(
        timeout=MOLTBOOK_TIMEOUT_SEC,
        connect=MOLTBOOK_TIMEOUT_SEC,
        read=MOLTBOOK_TIMEOUT_SEC,
        write=MOLTBOOK_TIMEOUT_SEC,
        pool=MOLTBOOK_TIMEOUT_SEC,
    )
    return httpx.Client(base_url=base_url, headers=headers, timeout=timeout)


def _request(method: str, path: str, params: Optional[dict] = None, payload: Optional[dict] = None) -> dict:
    try:
        with _client() as client:
            resp = client.request(method, path, params=params, json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException as exc:
        return {
            "ok": False,
            "error": "timeout",
            "method": method,
            "path": path,
            "detail": str(exc),
        }
    except httpx.HTTPStatusError as exc:
        preview = exc.response.text[:400] if exc.response is not None else ""
        status_code = exc.response.status_code if exc.response is not None else None
        return {
            "ok": False,
            "error": "http_status",
            "status_code": status_code,
            "method": method,
            "path": path,
            "detail": preview or str(exc),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": "request_failed",
            "method": method,
            "path": path,
            "detail": str(exc),
        }


@tool(
    description="Fetch Moltbook posts.",
    parameters={
        "type": "object",
        "properties": {
            "sort": {"type": "string", "enum": ["hot", "new"]},
            "limit": {"type": "integer"},
            "submolt": {"type": "string"},
        },
    },
)
@eidosian()
def moltbook_posts(sort: str = "new", limit: int = 20, submolt: Optional[str] = None) -> str:
    params: Dict[str, Any] = {"sort": sort, "limit": limit}
    if submolt:
        params["submolt"] = submolt
    data = _request("GET", "/posts", params=params)
    return json.dumps(data)


@tool(
    description="Fetch a Moltbook post by ID.",
    parameters={
        "type": "object",
        "properties": {"post_id": {"type": "string"}},
        "required": ["post_id"],
    },
)
@eidosian()
def moltbook_post(post_id: str) -> str:
    data = _request("GET", f"/posts/{post_id}")
    return json.dumps(data)


@tool(
    description="Fetch comments for a Moltbook post.",
    parameters={
        "type": "object",
        "properties": {"post_id": {"type": "string"}},
        "required": ["post_id"],
    },
)
@eidosian()
def moltbook_comments(post_id: str) -> str:
    data = _request("GET", f"/posts/{post_id}/comments")
    return json.dumps(data)


@tool(
    description="Create a Moltbook post.",
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "content": {"type": "string"},
            "submolt": {"type": "string"},
        },
        "required": ["title", "content"],
    },
)
@eidosian()
def moltbook_create(title: str, content: str, submolt: Optional[str] = None) -> str:
    payload: Dict[str, Any] = {"title": title, "content": content}
    if submolt:
        payload["submolt"] = submolt
    data = _request("POST", "/posts", payload=payload)
    return json.dumps(data)


@tool(
    description="Reply to a Moltbook post or comment.",
    parameters={
        "type": "object",
        "properties": {
            "post_id": {"type": "string"},
            "content": {"type": "string"},
            "parent_id": {"type": "string"},
        },
        "required": ["post_id", "content"],
    },
)
@eidosian()
def moltbook_reply(post_id: str, content: str, parent_id: Optional[str] = None) -> str:
    payload = {"content": content}
    if parent_id:
        payload["parent_id"] = parent_id
    data = _request("POST", f"/posts/{post_id}/comments", payload=payload)
    return json.dumps(data)


@tool(
    description="Upvote a Moltbook post.",
    parameters={
        "type": "object",
        "properties": {"post_id": {"type": "string"}},
        "required": ["post_id"],
    },
)
@eidosian()
def moltbook_upvote(post_id: str) -> str:
    data = _request("POST", f"/posts/{post_id}/upvote", payload={})
    return json.dumps(data)


@tool(
    description="Downvote a Moltbook post.",
    parameters={
        "type": "object",
        "properties": {"post_id": {"type": "string"}},
        "required": ["post_id"],
    },
)
@eidosian()
def moltbook_downvote(post_id: str) -> str:
    data = _request("POST", f"/posts/{post_id}/downvote", payload={})
    return json.dumps(data)


@tool(
    description="Upvote a Moltbook comment.",
    parameters={
        "type": "object",
        "properties": {"comment_id": {"type": "string"}},
        "required": ["comment_id"],
    },
)
@eidosian()
def moltbook_comment_upvote(comment_id: str) -> str:
    data = _request("POST", f"/comments/{comment_id}/upvote", payload={})
    return json.dumps(data)


@tool(
    description="Follow a Moltbook agent.",
    parameters={
        "type": "object",
        "properties": {"agent_name": {"type": "string"}},
        "required": ["agent_name"],
    },
)
@eidosian()
def moltbook_follow(agent_name: str) -> str:
    data = _request("POST", f"/agents/{agent_name}/follow", payload={})
    return json.dumps(data)


@tool(
    description="Unfollow a Moltbook agent.",
    parameters={
        "type": "object",
        "properties": {"agent_name": {"type": "string"}},
        "required": ["agent_name"],
    },
)
@eidosian()
def moltbook_unfollow(agent_name: str) -> str:
    data = _request("DELETE", f"/agents/{agent_name}/follow")
    return json.dumps(data)


@tool(
    description="Check Moltbook DM activity.",
    parameters={"type": "object", "properties": {}},
)
@eidosian()
def moltbook_dm_check() -> str:
    data = _request("GET", "/agents/dm/check")
    return json.dumps(data)


@tool(
    description="List Moltbook DM requests.",
    parameters={"type": "object", "properties": {}},
)
@eidosian()
def moltbook_dm_requests() -> str:
    data = _request("GET", "/agents/dm/requests")
    return json.dumps(data)


@tool(
    description="Send a Moltbook DM request.",
    parameters={
        "type": "object",
        "properties": {
            "target": {"type": "string"},
            "message": {"type": "string"},
            "to_owner": {"type": "boolean"},
        },
        "required": ["target", "message"],
    },
)
@eidosian()
def moltbook_dm_request(target: str, message: str, to_owner: bool = False) -> str:
    payload = {"message": message}
    if to_owner or target.startswith("@"):
        payload["to_owner"] = target.lstrip("@")
    else:
        payload["to"] = target
    data = _request("POST", "/agents/dm/request", payload=payload)
    return json.dumps(data)


@tool(
    description="Approve a Moltbook DM request.",
    parameters={
        "type": "object",
        "properties": {"conversation_id": {"type": "string"}},
        "required": ["conversation_id"],
    },
)
@eidosian()
def moltbook_dm_approve(conversation_id: str) -> str:
    data = _request("POST", f"/agents/dm/requests/{conversation_id}/approve", payload={})
    return json.dumps(data)


@tool(
    description="Reject a Moltbook DM request.",
    parameters={
        "type": "object",
        "properties": {
            "conversation_id": {"type": "string"},
            "block": {"type": "boolean"},
        },
        "required": ["conversation_id"],
    },
)
@eidosian()
def moltbook_dm_reject(conversation_id: str, block: bool = False) -> str:
    payload = {"block": True} if block else {}
    data = _request("POST", f"/agents/dm/requests/{conversation_id}/reject", payload=payload)
    return json.dumps(data)


@tool(
    description="List Moltbook DM conversations.",
    parameters={"type": "object", "properties": {}},
)
@eidosian()
def moltbook_dm_conversations() -> str:
    data = _request("GET", "/agents/dm/conversations")
    return json.dumps(data)


@tool(
    description="Read a Moltbook DM conversation.",
    parameters={
        "type": "object",
        "properties": {"conversation_id": {"type": "string"}},
        "required": ["conversation_id"],
    },
)
@eidosian()
def moltbook_dm_read(conversation_id: str) -> str:
    data = _request("GET", f"/agents/dm/conversations/{conversation_id}")
    return json.dumps(data)


@tool(
    description="Send a Moltbook DM message.",
    parameters={
        "type": "object",
        "properties": {
            "conversation_id": {"type": "string"},
            "message": {"type": "string"},
        },
        "required": ["conversation_id", "message"],
    },
)
@eidosian()
def moltbook_dm_send(conversation_id: str, message: str) -> str:
    payload = {"message": message}
    data = _request("POST", f"/agents/dm/conversations/{conversation_id}/send", payload=payload)
    return json.dumps(data)


@tool(
    description="Verify a Moltbook challenge response.",
    parameters={
        "type": "object",
        "properties": {
            "verification_code": {"type": "string"},
            "answer": {"type": "string"},
        },
        "required": ["verification_code", "answer"],
    },
)
@eidosian()
def moltbook_verify(verification_code: str, answer: str) -> str:
    payload = {"verification_code": verification_code, "answer": answer}
    data = _request("POST", "/verify", payload=payload)
    return json.dumps(data)


@tool(
    description="Fetch the authenticated Moltbook agent profile.",
    parameters={"type": "object", "properties": {}},
)
@eidosian()
def moltbook_me() -> str:
    data = _request("GET", "/agents/me")
    return json.dumps(data)


@tool(
    description="Check Moltbook agent status.",
    parameters={"type": "object", "properties": {}},
)
@eidosian()
def moltbook_status() -> str:
    data = _request("GET", "/agents/status")
    return json.dumps(data)


@tool(
    description="Fetch Moltbook feed.",
    parameters={
        "type": "object",
        "properties": {
            "sort": {"type": "string", "enum": ["hot", "new"]},
            "limit": {"type": "integer"},
        },
    },
)
@eidosian()
def moltbook_feed(sort: str = "new", limit: int = 15) -> str:
    params: Dict[str, Any] = {"sort": sort, "limit": limit}
    data = _request("GET", "/feed", params=params)
    return json.dumps(data)
