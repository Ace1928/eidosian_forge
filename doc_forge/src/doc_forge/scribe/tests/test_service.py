import subprocess
from unittest.mock import MagicMock, patch

import pytest
from doc_forge.scribe.service import DocProcessor, create_app
from fastapi.testclient import TestClient


@pytest.fixture
def mock_processor(mock_config):
    # Mock components to avoid real IO/Network
    with patch("doc_forge.scribe.service.ManagedModelServer"), patch(
        "doc_forge.scribe.service.DocumentExtractor"
    ), patch("doc_forge.scribe.service.DocGenerator"), patch("doc_forge.scribe.service.FederatedJudge"):

        proc = DocProcessor(mock_config)
        proc.extractor.extract.return_value = ("source", {})
        proc.generator.generate.return_value = "# Doc"
        proc.judges.evaluate.return_value = {"approved": True, "aggregate_score": 0.9}
        proc.model_server.is_ready.return_value = True

        yield proc


def test_processor_loop_lifecycle(mock_processor):
    # Create a dummy file to process
    p = mock_processor.cfg.source_root / "test.py"
    p.write_text("code", encoding="utf-8")

    # We want to run one iteration of the loop synchronously
    # Mock _scan_candidates to find our file
    mock_processor._scan_candidates = MagicMock(return_value=[p])

    # Control the loop: First call to stop_event.is_set() returns False (run),
    # Second call (inside pending loop) returns False (continue processing),
    # Third call (start of next while loop) returns True (stop)
    mock_processor.stop_event.is_set = MagicMock(side_effect=[False, False, True])

    # Avoid sleep
    with patch("time.sleep"):
        mock_processor._run_loop()

    assert mock_processor.state.get("status") == "stopped"
    assert mock_processor.state.get("total_discovered") == 1
    assert mock_processor.state.get("processed") == 1
    assert mock_processor.state.get("approved") == 1


def test_dashboard_redirect_uses_atlas_env(mock_config, monkeypatch):
    monkeypatch.setenv("EIDOS_ATLAS_URL", "http://127.0.0.1:9999/")
    processor = DocProcessor(mock_config)
    app = create_app(processor)
    with TestClient(app) as client:
        response = client.get("/", follow_redirects=False)
        status_response = client.get("/api/status")
    assert response.status_code == 307
    assert response.headers["location"] == "http://127.0.0.1:9999/"
    assert status_response.status_code == 200


def test_docs_api_endpoints(mock_config):
    root = mock_config.forge_root
    target = root / "doc_forge" / "src" / "doc_forge" / "scribe"
    target.mkdir(parents=True, exist_ok=True)
    (target / "service.py").write_text(
        'from fastapi import FastAPI\napp = FastAPI()\n@app.get("/health")\ndef health():\n    return {"ok": True}\n',
        encoding="utf-8",
    )
    (root / "cfg").mkdir(parents=True, exist_ok=True)
    (root / "cfg" / "documentation_policy.json").write_text(
        '{"documented_prefixes":["doc_forge"],"excluded_prefixes":[],"excluded_segments":[]}',
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=str(root), check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=str(root), check=True, capture_output=True, text=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(root), check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "."], cwd=str(root), check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(root), check=True, capture_output=True, text=True)

    processor = DocProcessor(mock_config)
    processor.start = MagicMock()
    processor.stop = MagicMock()
    app = create_app(processor)
    with TestClient(app) as client:
        coverage = client.get("/api/docs/coverage")
        assert coverage.status_code == 200
        payload = coverage.json()
        assert payload["contract"] == "eidos.documentation_inventory.v1"

        render = client.get("/api/docs/render", params={"path": "doc_forge/src/doc_forge/scribe"})
        assert render.status_code == 200
        assert "GET /health" in render.json()["content"]

        upsert = client.post("/api/docs/upsert", params={"path": "doc_forge/src/doc_forge/scribe"})
        assert upsert.status_code == 200
        assert upsert.json()["readme_path"] == "doc_forge/src/doc_forge/scribe/README.md"

        readme = client.get("/api/docs/readme", params={"path": "doc_forge/src/doc_forge/scribe"})
        assert readme.status_code == 200

        diff = client.get("/api/docs/diff", params={"path": "doc_forge/src/doc_forge/scribe"})
        assert diff.status_code == 200

        batch = client.post(
            "/api/docs/upsert-batch",
            params={"path_prefix": "doc_forge/src/doc_forge", "limit": 5, "missing_only": False},
        )
        assert batch.status_code == 200
        batch_payload = batch.json()
        assert batch_payload["contract"] == "eidos.docs_upsert_batch.v1"
        assert batch_payload["write_count"] >= 1
