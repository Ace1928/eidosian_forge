from __future__ import annotations

import json
from pathlib import Path

from eidos_mcp.routers import code


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_code_forge_provenance_prefers_registry_payload(tmp_path: Path, monkeypatch) -> None:
    forge_root = tmp_path / "forge"
    record_dir = forge_root / "data" / "code_forge" / "cycle" / "run_001"

    _write_json(
        record_dir / "provenance_links.json",
        {
            "generated_at": "2026-02-21T00:00:00+00:00",
            "stage": "archive_digester",
            "root_path": str((tmp_path / "repo").resolve()),
            "provenance_id": "links_only",
            "integration_policy": "run",
            "integration_run_id": "r1",
            "artifacts": [],
            "knowledge_links": {"count": 1},
            "memory_links": {"count": 1},
            "graphrag_links": {"count": 1},
        },
    )
    _write_json(
        record_dir / "provenance_registry.json",
        {
            "schema_version": "code_forge_provenance_registry_v1",
            "generated_at": "2026-02-21T01:00:00+00:00",
            "registry_id": "reg_1",
            "stage": "archive_digester",
            "root_path": str((tmp_path / "repo").resolve()),
            "provenance_id": "prov_1",
            "integration_policy": "effective_run",
            "integration_run_id": "r1",
            "artifacts": [{"artifact_kind": "triage"}],
            "links": {
                "knowledge_count": 3,
                "memory_count": 2,
                "graphrag_count": 4,
                "unit_links": [
                    {
                        "unit_id": "u1",
                        "knowledge_node_id": "n1",
                        "memory_id": "m1",
                        "qualified_name": "pkg.mod.f",
                    }
                ],
            },
            "benchmark": {"gate_pass": True},
        },
    )

    monkeypatch.setattr(code, "FORGE_DIR", forge_root.resolve())
    payload = json.loads(code.code_forge_provenance(include_unit_links=True, include_benchmark=True))
    assert payload["count"] == 1
    record = payload["records"][0]
    assert record["schema_version"] == "code_forge_provenance_registry_v1"
    assert record["knowledge_link_count"] == 3
    assert record["memory_link_count"] == 2
    assert record["graphrag_link_count"] == 4
    assert record["unit_link_count"] == 1
    assert record["unit_links"][0]["unit_id"] == "u1"
    assert record["benchmark"]["gate_pass"] is True


def test_code_forge_provenance_unit_filter(tmp_path: Path, monkeypatch) -> None:
    forge_root = tmp_path / "forge"
    record_dir = forge_root / "data" / "code_forge" / "cycle" / "run_002"
    _write_json(
        record_dir / "provenance_registry.json",
        {
            "schema_version": "code_forge_provenance_registry_v1",
            "generated_at": "2026-02-21T01:00:00+00:00",
            "registry_id": "reg_2",
            "stage": "roundtrip",
            "root_path": str((tmp_path / "repo").resolve()),
            "provenance_id": "prov_2",
            "integration_policy": "global",
            "integration_run_id": None,
            "artifacts": [],
            "links": {
                "knowledge_count": 0,
                "memory_count": 0,
                "graphrag_count": 0,
                "unit_links": [{"unit_id": "u-keep"}, {"unit_id": "u-other"}],
            },
        },
    )
    monkeypatch.setattr(code, "FORGE_DIR", forge_root.resolve())

    filtered = json.loads(code.code_forge_provenance(unit_id="u-keep", include_unit_links=True))
    assert filtered["count"] == 1
    rec = filtered["records"][0]
    assert rec["unit_filter"] == "u-keep"
    assert rec["unit_link_match_count"] == 1
    assert rec["unit_links"][0]["unit_id"] == "u-keep"

    empty = json.loads(code.code_forge_provenance(unit_id="missing"))
    assert empty["count"] == 0
