#!/usr/bin/env python3
"""
Tests for utility functions.
"""

import io
import sys
import unittest
from typing import Any, Dict
from unittest.mock import Mock, patch, MagicMock

from helpers.common import (
        DEFAULT_OLLAMA_API_URL,
        make_api_request,
        print_error,
        print_header,
        print_info,
        print_success,
        print_warning,
        print_json,
        check_ollama_running,
        ensure_ollama_running,
        check_ollama_installed,
    )
from helpers.install_ollama import (
        check_ollama_installed,
        ensure_ollama_running,
        check_ollama_running,
    )

class TestHelpers(unittest.TestCase):
    """Test cases for utility functions."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Redirect stdout to capture printed output
        self.captured_output = io.StringIO()
        self.original_stdout = sys.stdout
        sys.stdout = self.captured_output

    def tearDown(self) -> None:
        """Tear down test fixtures."""
        # Reset stdout
        sys.stdout = self.original_stdout

    def test_print_functions(self) -> None:
        """Test the print helper functions."""
        test_message = "Test message"

        print_header(test_message)
        print_success(test_message)
        print_error(test_message)
        print_info(test_message)
        print_warning(test_message)
        print_json({"key": "value"})

        output = self.captured_output.getvalue()

        # Check that the output contains all messages
        self.assertIn(test_message, output)
        # The output should have 5 occurrences of the test message
        self.assertEqual(output.count(test_message), 5)

        # Updated assertions to match actual output format
        self.assertIn("===", output)  # Header format
        self.assertIn("✓", output)  # Success symbol
        self.assertIn("✗", output)  # Error symbol
        self.assertIn("ℹ", output)  # Info symbol
        self.assertIn("⚠", output)  # Warning symbol

    @patch("requests.Session")
    def test_make_api_request_success(self, mock_session: Any) -> None:
        """Test successful API requests."""
        # Setup mock session and response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"version": "1.0.0"}
        
        # Set up the mock session instance
        session_instance = mock_session.return_value
        session_instance.request.return_value = mock_response

        # Call function
        result = make_api_request(
            "GET", "/api/version", base_url=DEFAULT_OLLAMA_API_URL
        )

        # Assert results
        session_instance.request.assert_called_once()
        self.assertEqual(result, {"version": "1.0.0"})

    @patch("requests.Session")
    def test_make_api_request_with_data(self, mock_session: Any) -> None:
        """Test API requests with data payload."""
        # Setup mock session and response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"response": "generated text"}
        
        # Set up the mock session instance
        session_instance = mock_session.return_value
        session_instance.request.return_value = mock_response

        # Test data
        test_data: Dict[str, Any] = {"model": "test-model", "prompt": "test prompt"}

        # Call function
        result = make_api_request(
            "POST", "/api/generate", data=test_data, base_url=DEFAULT_OLLAMA_API_URL
        )

        # Assert results
        session_instance.request.assert_called_once()
        self.assertEqual(result, {"response": "generated text"})

    @patch("requests.Session")
    def test_make_api_request(self, mock_session):
        """Test make_api_request function."""
        # Setup mock session and response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"result": "success"}
        
        # Set up the mock session instance
        session_instance = mock_session.return_value
        session_instance.request.return_value = mock_response
        
        # Call function
        result = make_api_request("GET", "/test")
        
        # Verify result
        self.assertEqual(result, {"result": "success"})
        session_instance.request.assert_called_once()
    
    @patch("helpers.common.subprocess.run")
    def test_check_ollama_running(self, mock_run):
        """Test check_ollama_running function."""
        # Setup mock for running Ollama
        mock_run.return_value.returncode = 0
        
        # Call function
        is_running, message = check_ollama_running()
        
        # Verify result
        self.assertTrue(is_running)
        self.assertIn("Ollama server is running", message)

    def test_check_ollama_installed(self):
        self.assertTrue(check_ollama_installed())

    def test_ensure_ollama_running(self):
        is_running, message = ensure_ollama_running()
        self.assertTrue(is_running)
        self.assertIn("Ollama is ready", message)
        self.assertTrue(message)
        
if __name__ == "__main__":
        unittest.main()
