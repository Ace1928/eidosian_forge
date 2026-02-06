from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def test_security_headers_present(monkeypatch) -> None:
    monkeypatch.setenv("MOLTBOOK_ALLOWED_HOSTS", "testserver")
    monkeypatch.setenv("MOLTBOOK_CSP_ENABLE", "true")
    monkeypatch.setenv("MOLTBOOK_MOCK", "true")
    from moltbook_forge.ui import app as app_module

    importlib.reload(app_module)
    with TestClient(app_module.app) as client:
        response = client.get("/")
        assert response.status_code == 200
        headers = response.headers
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["x-frame-options"] == "DENY"
        assert headers["referrer-policy"] == "same-origin"
        assert "content-security-policy" in headers
        assert "frame-ancestors 'none'" in headers["content-security-policy"]


def test_accessibility_landmarks(monkeypatch) -> None:
    monkeypatch.setenv("MOLTBOOK_ALLOWED_HOSTS", "testserver")
    monkeypatch.setenv("MOLTBOOK_MOCK", "true")
    from moltbook_forge.ui import app as app_module

    importlib.reload(app_module)
    with TestClient(app_module.app) as client:
        response = client.get("/")
        assert response.status_code == 200
        body = response.text
        assert "Skip to main content" in body
        assert 'role="main"' in body
        assert 'id="main"' in body
        assert 'aria-live="polite"' in body
        assert 'tabindex="-1"' in body
