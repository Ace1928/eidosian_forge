import pytest
import json
import logging
from pathlib import Path
from diagnostics_forge import DiagnosticsForge

def test_diag_logging(tmp_path):
    diag = DiagnosticsForge(log_dir=tmp_path, service_name="test_service")
    diag.log_event("INFO", "Test message", user="eidos")
    
    log_file = tmp_path / "test_service.log"
    assert log_file.exists()
    content = log_file.read_text()
    assert "Test message" in content
    assert 'DATA: {"user": "eidos"}' in content

def test_diag_json_logging(tmp_path):
    diag = DiagnosticsForge(log_dir=tmp_path, service_name="test_json", json_format=True)
    diag.log_event("INFO", "Json test", value=42)
    
    log_file = tmp_path / "test_json.log"
    content = log_file.read_text()
    # Parse last line
    lines = content.strip().split('\n')
    last_line = json.loads(lines[-1])
    
    assert last_line["message"] == "Json test"
    assert last_line["value"] == 42
    assert last_line["level"] == "INFO"

def test_diag_metrics(tmp_path):
    diag = DiagnosticsForge(log_dir=tmp_path, service_name="test_metrics")
    t = diag.start_timer("foo")
    diag.stop_timer(t)
    
    summary = diag.get_metrics_summary("foo")
    assert summary["count"] == 1
    
    diag.save_metrics()
    assert (tmp_path / "test_metrics_metrics.json").exists()

def test_log_rotation(tmp_path):
    # Set small max_bytes
    diag = DiagnosticsForge(log_dir=tmp_path, service_name="test_rot", max_bytes=100)
    
    # Write enough to rotate
    for i in range(10):
        diag.log_event("INFO", "A" * 50) # 50 chars + overhead > 100 bytes eventually
    
    # Check for rotated files
    assert (tmp_path / "test_rot.log").exists()
    assert (tmp_path / "test_rot.log.1").exists()
