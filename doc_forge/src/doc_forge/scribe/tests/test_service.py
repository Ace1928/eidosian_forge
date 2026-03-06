from unittest.mock import MagicMock, patch

import pytest
from doc_forge.scribe.generate import REQUIRED_HEADINGS
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


def test_process_pending_returns_summary(mock_processor):
    p = mock_processor.cfg.source_root / "sample.py"
    p.write_text("print('hi')", encoding="utf-8")
    mock_processor._scan_candidates = MagicMock(return_value=[p])
    mock_processor._queue_lexicon_terms = MagicMock(return_value={"queued": True})
    mock_processor.generator.generate.return_value = "\n".join(
        [
            "# File Overview",
            "## Key Structures",
            "## Behavior Summary",
            "## Validation Notes",
            "## Improvement Opportunities",
        ]
    )
    mock_processor.judges.evaluate.return_value = {"approved": True, "aggregate_score": 0.91}

    summary = mock_processor.process_pending(max_documents=1)

    assert summary["processed"] == 1
    assert summary["approved"] == 1
    assert summary["documents"][0]["approved"] is True
    assert summary["documents"][0]["lexicon_queue"]["queued"] is True


def test_process_pending_skips_managed_server_when_disabled(mock_processor):
    p = mock_processor.cfg.source_root / "contract.py"
    p.write_text("print('hi')", encoding="utf-8")
    mock_processor.cfg.dry_run = False
    mock_processor.cfg.enable_managed_llm = False
    mock_processor._scan_candidates = MagicMock(return_value=[p])
    mock_processor._queue_lexicon_terms = MagicMock(return_value={"queued": True})
    mock_processor.generator.generate.return_value = "\n".join(REQUIRED_HEADINGS) + "\nContract path"
    mock_processor.judges.evaluate.return_value = {"approved": True, "aggregate_score": 0.91}

    summary = mock_processor.process_pending(max_documents=1)

    assert summary["processed"] == 1
    mock_processor.model_server.start.assert_not_called()


def test_scan_candidates_allows_scoped_source_root_under_runtime(mock_config):
    scoped_root = mock_config.forge_root / "data" / "runtime" / "validation" / "source"
    scoped_root.mkdir(parents=True, exist_ok=True)
    file_path = scoped_root / "module.py"
    file_path.write_text("print('ok')", encoding="utf-8")
    cfg = mock_config.__class__(**{**mock_config.__dict__, "source_root": scoped_root})

    with patch("doc_forge.scribe.service.ManagedModelServer"), patch(
        "doc_forge.scribe.service.DocumentExtractor"
    ), patch("doc_forge.scribe.service.DocGenerator"), patch("doc_forge.scribe.service.FederatedJudge"):
        proc = DocProcessor(cfg)

    candidates = proc._scan_candidates()

    assert file_path in candidates
