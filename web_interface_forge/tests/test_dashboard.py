from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from web_interface_forge.dashboard import main as dashboard


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_doc_status_api_and_index_page(monkeypatch, tmp_path: Path) -> None:
    runtime = tmp_path / "doc_forge" / "runtime"
    final_docs = runtime / "final_docs"
    final_docs.mkdir(parents=True, exist_ok=True)

    _write_json(
        runtime / "processor_status.json",
        {
            "status": "running",
            "processed": 12,
            "remaining": 4,
            "average_quality_score": 0.88,
            "last_approved": "foo/bar.py",
        },
    )
    _write_json(
        runtime / "doc_index.json",
        {
            "entries": [
                {
                    "source": "foo/bar.py",
                    "document": "foo/bar.py.md",
                    "score": 0.91,
                    "doc_type": "py",
                    "updated_at": "2026-02-26T00:00:00+00:00",
                }
            ]
        },
    )

    (final_docs / "foo").mkdir(parents=True, exist_ok=True)
    (final_docs / "foo" / "bar.py.md").write_text("# Example\n", encoding="utf-8")

    monkeypatch.setattr(dashboard, "DOC_RUNTIME", runtime)
    monkeypatch.setattr(dashboard, "DOC_FINAL", final_docs)
    monkeypatch.setattr(dashboard, "DOC_INDEX", runtime / "doc_index.json")
    monkeypatch.setattr(dashboard, "DOC_STATUS", runtime / "processor_status.json")

    with TestClient(dashboard.app) as client:
        resp = client.get("/api/doc/status")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["index_count"] == 1
        assert payload["status"]["processed"] == 12

        html = client.get("/").text
        assert "foo/bar.py" in html
        assert "Indexed Docs" in html


def test_browse_blocks_path_traversal() -> None:
    with TestClient(dashboard.app) as client:
        resp = client.get("/browse/%2e%2e/%2e%2e/etc/passwd")
        assert resp.status_code == 403


def test_health_endpoint() -> None:
    with TestClient(dashboard.app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "ok"
