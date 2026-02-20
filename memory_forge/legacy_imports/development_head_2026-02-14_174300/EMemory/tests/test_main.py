#!/usr/bin/env python3
"""
Test suite for main.py functionality.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import the local main.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now import from the local main.py
from main import (
    check_system,
    parse_arguments,
    run_tests,
)


class TestMainModule(unittest.TestCase):
    """Test cases for the main entry point of the EidosMemorySystem."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary log directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_argv = sys.argv.copy()

    def tearDown(self):
        """Clean up after tests."""
        sys.argv = self.original_argv
        self.temp_dir.cleanup()

    def test_parse_arguments(self):
        """Test command line argument parsing."""
        # Test with default arguments
        sys.argv = ["main.py"]
        args = parse_arguments()
        self.assertFalse(args.skip_tests)
        self.assertFalse(args.verbose)
        self.assertIsNone(args.demo)
        self.assertIsNone(args.batch)

        # Test with explicit arguments
        sys.argv = [
            "main.py",
            "--skip-tests",
            "--verbose",
            "--demo",
            "basic",
            "--model",
            "test-model",
            "--backend",
            "ollama",
        ]
        args = parse_arguments()
        self.assertTrue(args.skip_tests)
        self.assertTrue(args.verbose)
        self.assertEqual(args.demo, "basic")
        self.assertEqual(args.model, "test-model")
        self.assertEqual(args.backend, "ollama")

    @patch("main.Path")
    @patch("requests.get")
    def test_check_system(self, mock_get, mock_path):
        """Test system check functionality."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Mock file operations
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.parent.mkdir.return_value = None
        mock_path_instance.write_text.return_value = None
        mock_path_instance.unlink.return_value = None

        success, issues = check_system()
        self.assertTrue(success)
        self.assertEqual(len(issues), 0)

        # Test failure case
        mock_get.side_effect = Exception("Connection error")
        success, issues = check_system()
        self.assertFalse(success)
        self.assertGreater(len(issues), 0)

    @patch("subprocess.run")
    def test_run_tests(self, mock_run):
        """Test running unit tests."""
        # Mock successful test run
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "All tests passed"
        mock_run.return_value = mock_process

        result = run_tests()
        self.assertTrue(result)
        mock_run.assert_called_once()

        # Test failure case
        mock_process.returncode = 1
        mock_process.stderr = "Test failures occurred"


if __name__ == "__main__":
    unittest.main()
