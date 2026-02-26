import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from doc_forge.scribe.service import DocProcessor, create_app

@pytest.fixture
def mock_processor(mock_config):
    # Mock components to avoid real IO/Network
    with patch("doc_forge.scribe.service.ManagedModelServer"), \
         patch("doc_forge.scribe.service.DocumentExtractor"), \
         patch("doc_forge.scribe.service.DocGenerator"), \
         patch("doc_forge.scribe.service.FederatedJudge"):
        
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
