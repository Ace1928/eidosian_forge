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
    bridge_payload = bridge.json()
    assert bridge_payload["contract"] == "eidos.word_forge.bridge.summary.v1"
    assert "community_summary" in bridge_payload

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


def test_atlas_word_graph_community_route_returns_payload() -> None:
    module = import_module("atlas_forge.app")
    client = TestClient(module.app)

    response = client.get("/api/graph/word/communities")
    assert response.status_code == 200
    payload = response.json()
    assert payload["contract"] == "eidos.atlas.word_graph.communities.v1"
    assert "top_communities" in payload
