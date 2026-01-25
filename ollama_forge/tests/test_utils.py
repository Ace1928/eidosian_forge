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

        # Assertions match actual symbols used in common.py
        self.assertIn("===", output)  # Header format
        self.assertIn("✅", output)   # Success symbol
        self.assertIn("❌", output)   # Error symbol
        self.assertIn("ℹ️", output)   # Info symbol
        self.assertIn("⚠️", output)   # Warning symbol

    @patch("helpers.common.httpx.Client")
    def test_make_api_request_success(self, mock_client_class: Any) -> None:
        """Test successful API requests."""
        # Setup mock client and response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"version": "1.0.0"}
        
        # Set up the mock client instance
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        # Call function - note: endpoint is first arg in actual implementation
        result = make_api_request("/api/version")

        # Assert results
        mock_client.get.assert_called_once()
        self.assertEqual(result, {"version": "1.0.0"})

    @patch("helpers.common.httpx.Client")
    def test_make_api_request_with_data(self, mock_client_class: Any) -> None:
        """Test API requests with data payload."""
        # Setup mock client and response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"response": "generated text"}
        
        # Set up the mock client instance
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        # Test data
        test_data: Dict[str, Any] = {"model": "test-model", "prompt": "test prompt"}

        # Call function - endpoint first, method second, data third
        result = make_api_request("/api/generate", method="POST", data=test_data)

        # Assert results
        mock_client.post.assert_called_once()
        self.assertEqual(result, {"response": "generated text"})

    @patch("helpers.common.httpx.Client")
    def test_make_api_request_get(self, mock_client_class):
        """Test make_api_request function with GET."""
        # Setup mock client and response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"result": "success"}
        
        # Set up the mock client instance
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client
        
        # Call function
        result = make_api_request("/test")
        
        # Verify result
        self.assertEqual(result, {"result": "success"})
        mock_client.get.assert_called_once()
    
    @patch("helpers.common.httpx.Client")
    def test_check_ollama_running(self, mock_client_class):
        """Test check_ollama_running function."""
        # Setup mock for running Ollama
        mock_response = Mock()
        mock_response.status_code = 200
        
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client
        
        # Call function - returns bool only, not tuple
        is_running = check_ollama_running()
        
        # Verify result
        self.assertTrue(is_running)

    def test_check_ollama_installed(self):
        """Test check_ollama_installed function - real check."""
        # This is an actual check - may pass or fail depending on system
        result = check_ollama_installed()
        self.assertIsInstance(result, bool)

    @patch("helpers.common.httpx.Client")
    def test_ensure_ollama_running(self, mock_client_class):
        """Test ensure_ollama_running function."""
        # Setup mock for running Ollama
        mock_response = Mock()
        mock_response.status_code = 200
        
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client
        
        # Call function - returns bool only, not tuple
        is_running = ensure_ollama_running()
        
        # Verify result
        self.assertTrue(is_running)
        
        
if __name__ == "__main__":
    unittest.main()
