from __future__ import annotations

import json
import sys
import types

import httpx


def _install_mcp_stub() -> None:
    try:
        from mcp.server.fastmcp import FastMCP  # noqa: F401

        return
    except Exception:
        pass

    if "mcp.server.fastmcp" in sys.modules:
        return
    mcp_module = types.ModuleType("mcp")
    server_module = types.ModuleType("mcp.server")
    fastmcp_module = types.ModuleType("mcp.server.fastmcp")

    class _FastMCP:  # pragma: no cover - shim for tests
        def __init__(self, *args, **kwargs):
            pass

        def tool(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    fastmcp_module.FastMCP = _FastMCP
    server_module.fastmcp = fastmcp_module
    mcp_module.server = server_module

    sys.modules["mcp"] = mcp_module
    sys.modules["mcp.server"] = server_module
    sys.modules["mcp.server.fastmcp"] = fastmcp_module


_install_mcp_stub()

from eidos_mcp.routers import moltbook  # noqa: E402


class _MockResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _MockClient:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def request(self, method, path, params=None, json=None):
        return _MockResponse({"method": method, "path": path, "params": params, "payload": json, **self.payload})


class _TimeoutClient:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def request(self, method, path, params=None, json=None):
        raise httpx.TimeoutException("timed out")


class _ErrorResponse:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text
        self.request = httpx.Request("GET", "https://www.moltbook.com/api/v1/test")

    def raise_for_status(self) -> None:
        raise httpx.HTTPStatusError(
            "boom",
            request=self.request,
            response=httpx.Response(self.status_code, request=self.request, text=self.text),
        )

    def json(self) -> dict:
        return {}


class _ErrorClient:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def request(self, method, path, params=None, json=None):
        return _ErrorResponse(500, '{"success":false}')


def test_moltbook_posts(monkeypatch):
    monkeypatch.setattr(moltbook, "_client", lambda: _MockClient({"posts": []}))
    data = json.loads(moltbook.moltbook_posts(sort="hot", limit=5))
    assert data["method"] == "GET"
    assert data["path"] == "/posts"
    assert data["params"]["sort"] == "hot"
    assert data["params"]["limit"] == 5


def test_moltbook_reply_and_verify(monkeypatch):
    monkeypatch.setattr(moltbook, "_client", lambda: _MockClient({"ok": True}))
    reply = json.loads(moltbook.moltbook_reply("post-123", "hello", parent_id="comment-1"))
    assert reply["method"] == "POST"
    assert reply["path"] == "/posts/post-123/comments"
    assert reply["payload"]["content"] == "hello"
    assert reply["payload"]["parent_id"] == "comment-1"

    verify = json.loads(moltbook.moltbook_verify("code-1", "42.00"))
    assert verify["path"] == "/verify"
    assert verify["payload"]["verification_code"] == "code-1"
    assert verify["payload"]["answer"] == "42.00"


def test_moltbook_dm_request(monkeypatch):
    monkeypatch.setattr(moltbook, "_client", lambda: _MockClient({"ok": True}))
    res = json.loads(moltbook.moltbook_dm_request("AgentX", "Hello"))
    assert res["path"] == "/agents/dm/request"
    assert res["payload"]["to"] == "AgentX"

    res_owner = json.loads(moltbook.moltbook_dm_request("@owner", "Hi", to_owner=True))
    assert res_owner["payload"]["to_owner"] == "owner"


def test_moltbook_timeout_and_http_errors(monkeypatch):
    monkeypatch.setattr(moltbook, "_client", lambda: _TimeoutClient())
    timeout_res = json.loads(moltbook.moltbook_status())
    assert timeout_res["ok"] is False
    assert timeout_res["error"] == "timeout"

    monkeypatch.setattr(moltbook, "_client", lambda: _ErrorClient())
    status_res = json.loads(moltbook.moltbook_status())
    assert status_res["ok"] is False
    assert status_res["error"] == "http_status"
    assert status_res["status_code"] == 500
