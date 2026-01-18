import unittest
import os
import shutil
import json
from pathlib import Path
from eidosian_forge.diagnostics_forge import DiagnosticsForge

class TestDiagnosticsForge(unittest.TestCase):
    def setUp(self):
        self.test_log_dir = Path("test_logs")
        self.diag = DiagnosticsForge(log_dir=str(self.test_log_dir), service_name="test_service")

    def tearDown(self):
        if self.test_log_dir.exists():
            shutil.rmtree(self.test_log_dir)

    def test_log_creation(self):
        self.diag.log_event("INFO", "Test message")
        # Check if log file exists
        log_files = list(self.test_log_dir.glob("test_service_*.log"))
        self.assertTrue(len(log_files) > 0)
        
        with open(log_files[0], 'r') as f:
            content = f.read()
            self.assertIn("Test message", content)

    def test_metrics_timing(self):
        timer = self.diag.start_timer("computation")
        # Simulate work
        import time
        time.sleep(0.1)
        duration = self.diag.stop_timer(timer)
        
        self.assertGreaterEqual(duration, 0.1)
        summary = self.diag.get_metrics_summary("computation")
        self.assertEqual(summary["count"], 1)
        self.assertGreaterEqual(summary["avg"], 0.1)

    def test_save_metrics(self):
        timer = self.diag.start_timer("test")
        self.diag.stop_timer(timer)
        
        metrics_file = self.test_log_dir / "test_metrics.json"
        self.diag.save_metrics(str(metrics_file))
        
        self.assertTrue(metrics_file.exists())
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            self.assertIn("test", data)

if __name__ == "__main__":
    unittest.main()
