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


def test_atlas_word_forge_routes_return_payloads() -> None:
    module = import_module("atlas_forge.app")
    client = TestClient(module.app)

    multilingual = client.get("/api/word-forge/multilingual")
    assert multilingual.status_code == 200
    assert multilingual.json()["contract"] == "eidos.word_forge.multilingual.summary.v1"

    bridge = client.get("/api/word-forge/bridge-audit")
    assert bridge.status_code == 200
    assert bridge.json()["contract"] == "eidos.word_forge.bridge.summary.v1"

    services = client.get("/api/runtime/services")
    assert services.status_code == 200
    assert services.json()["contract"] == "eidos.runtime.services.v1"


def test_atlas_word_graph_route_supports_multilingual_nodes() -> None:
    module = import_module("atlas_forge.app")
    client = TestClient(module.app)

    response = client.get("/api/graph/word")
    assert response.status_code == 200
    payload = response.json()
    assert "nodes" in payload
    assert "edges" in payload
