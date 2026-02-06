from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def _client(monkeypatch) -> TestClient:
    monkeypatch.setenv("MOLTBOOK_ALLOWED_HOSTS", "testserver")
    monkeypatch.setenv("MOLTBOOK_MOCK", "true")
    from moltbook_forge.ui import app as app_module

    importlib.reload(app_module)
    return TestClient(app_module.app)


def test_detail_push_url_present(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        response = client.get("/")
        assert response.status_code == 200
        body = response.text
        assert "hx-push-url" in body
        assert "/?selected=" in body


def test_selected_autoload_detail(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        response = client.get("/?selected=mock-post-1")
        assert response.status_code == 200
        assert 'hx-get="/api/detail/mock-post-1"' in response.text
