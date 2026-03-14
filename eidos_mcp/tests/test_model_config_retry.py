from __future__ import annotations

import httpx
from eidos_mcp.config.models import ModelConfig


def test_generate_payload_retries_remote_protocol_errors(monkeypatch) -> None:
    calls = {"count": 0}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json):
            calls["count"] += 1
            if calls["count"] < 3:
                raise httpx.RemoteProtocolError("server disconnected")
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={"response": "READY"},
            )

    monkeypatch.setattr(httpx, "Client", _Client)
    cfg = ModelConfig()

    payload = cfg.generate_payload("hello", timeout=1.0)

    assert payload["response"] == "READY"
    assert calls["count"] == 3


def test_generate_payload_retries_transient_http_status(monkeypatch) -> None:
    calls = {"count": 0}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json):
            calls["count"] += 1
            if calls["count"] < 2:
                return httpx.Response(500, request=httpx.Request("POST", url), json={"error": "boom"})
            return httpx.Response(200, request=httpx.Request("POST", url), json={"response": "OK"})

    monkeypatch.setattr(httpx, "Client", _Client)
    cfg = ModelConfig()

    payload = cfg.generate_payload("hello", timeout=1.0)

    assert payload["response"] == "OK"
    assert calls["count"] == 2
