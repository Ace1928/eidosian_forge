from __future__ import annotations

import json
from pathlib import Path

from eidos_mcp import eidos_mcp_server as server


def test_resource_context_index_missing_returns_json_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(server, "FORGE_DIR", tmp_path)
    monkeypatch.setattr(server, "ROOT_DIR", tmp_path)

    payload = server.resource_context_index()
    data = json.loads(payload)

    assert data["error"] == "Resource not found"
    assert data["resource"] == "eidos://context/index"
    assert isinstance(data["searched_paths"], list)
    assert data["searched_paths"]


def test_resource_todo_prefers_existing_source(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    forge = tmp_path / "forge"
    root.mkdir(parents=True)
    forge.mkdir(parents=True)

    todo_path = root / "TODO.md"
    todo_path.write_text("TODO marker\n", encoding="utf-8")

    monkeypatch.setattr(server, "ROOT_DIR", root)
    monkeypatch.setattr(server, "FORGE_DIR", forge)

    content = server.resource_todo()
    assert "TODO marker" in content


def test_resource_config_is_json() -> None:
    payload = server.resource_config()
    data = json.loads(payload)
    assert isinstance(data, dict)
