from __future__ import annotations

from fastapi.testclient import TestClient
from importlib import import_module


def test_atlas_app_basic_routes() -> None:
    module = import_module("atlas_forge.app")
    client = TestClient(module.app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    home = client.get("/")
    assert home.status_code == 200

    browse = client.get("/browse/")
    assert browse.status_code == 200
