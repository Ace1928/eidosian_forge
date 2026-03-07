from __future__ import annotations

from scripts.ci.changed_manifest import build_manifest


def test_manifest_filters_runtime_and_external_reference_paths() -> None:
    payload = build_manifest(
        [
            "data/runtime/live_doc_validation/source/eidos_mcp/src/eidos_mcp/routers/word_forge.py",
            "docs/external_references/2026-03-07-master-program/tika/tika_server.html",
            "agent_forge/src/agent_forge/autonomy/supervisor.py",
        ]
    )
    assert payload["python_files"] == ["agent_forge/src/agent_forge/autonomy/supervisor.py"]
    assert payload["has_python_files"] is True


def test_manifest_selects_changed_python_components_only() -> None:
    payload = build_manifest(
        [
            "agent_forge/src/agent_forge/autonomy/supervisor.py",
            "scripts/eidos_scheduler.py",
            "game_forge/src/autoseed/src/main.ts",
        ]
    )
    include = payload["python_components"]["include"]
    assert [entry["component"] for entry in include] == ["agent_forge", "scripts"]
    ts_include = payload["typescript_projects"]["include"]
    assert ts_include == [{"project": "autoseed", "root": "game_forge/src/autoseed"}]


def test_manifest_expands_to_all_components_for_global_python_changes() -> None:
    payload = build_manifest([".github/workflows/ci.yml"])
    include = payload["python_components"]["include"]
    components = [entry["component"] for entry in include]
    assert "agent_forge" in components
    assert "web_interface_forge" in components
    assert payload["has_python_files"] is False
