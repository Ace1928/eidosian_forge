#!/usr/bin/env python3
"""
Tests for the chat functionality.
"""

import json
import unittest
from typing import Any
from unittest.mock import Mock, patch, MagicMock

from helpers.model_constants import DEFAULT_CHAT_MODEL
from ollama_forge import OllamaClient


class TestChat(unittest.TestCase):
    """Test cases for chat functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.test_messages = [{"role": "user", "content": "Hello!"}]
        self.client = OllamaClient()

    @patch("ollama_forge.client.httpx.Client")
    def test_chat_function(self, mock_client_class: Any) -> None:
        """Test the chat function with non-streaming response."""
        # Setup mock response matching actual Ollama chat API format
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "model": "phi3:mini",
            "message": {"role": "assistant", "content": "Hello there!"},
            "done": True,
        }
        
        # Set up the mock client instance
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        # Call function with test messages
        result = self.client.chat(
            model=DEFAULT_CHAT_MODEL,
            messages=self.test_messages.copy()
        )

        # Assert results
        mock_client.post.assert_called_once()
        self.assertEqual(result["message"]["content"], "Hello there!")
        self.assertEqual(len(self.test_messages), 1)  # Original list unchanged

    @patch("ollama_forge.client.httpx.Client")
    def test_chat_with_options(self, mock_client_class: Any) -> None:
        """Test chat with temperature and other options."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "model": "phi3:mini",
            "message": {"role": "assistant", "content": "Options test response"},
            "done": True,
        }
        
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        # Call function with options
        result = self.client.chat(
            model=DEFAULT_CHAT_MODEL,
            messages=self.test_messages.copy(),
            options={"temperature": 0.5}
        )

        # Assert results
        self.assertIsNotNone(result)
        self.assertEqual(result["message"]["content"], "Options test response")

    @patch("ollama_forge.client.httpx.Client")
    def test_chat_error_handling(self, mock_client_class: Any) -> None:
        """Test chat error handling."""
        # Setup mock to raise an error
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("Connection error")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        # Call should raise the exception
        with self.assertRaises(Exception):
            self.client.chat(
                model=DEFAULT_CHAT_MODEL,
                messages=self.test_messages.copy()
            )

    @patch("ollama_forge.client.httpx.Client")
    def test_chat_streaming_not_implemented(self, mock_client_class: Any) -> None:
        """Test that streaming raises NotImplementedError."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        # Streaming should raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.client.chat(
                model=DEFAULT_CHAT_MODEL,
                messages=self.test_messages.copy(),
                stream=True
            )

    def test_initialize_chat(self):
        """Test the initialize_chat helper function."""
        from examples.chat_example import initialize_chat
        
        # Test with system message
        system_msg = "You are a helpful AI."
        messages = initialize_chat(DEFAULT_CHAT_MODEL, system_msg)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], system_msg)

        # Test without system message
        messages = initialize_chat(DEFAULT_CHAT_MODEL)
        self.assertEqual(len(messages), 0)

    @patch("ollama_forge.client.httpx.Client")
    def test_chat_response_format(self, mock_client_class: Any) -> None:
        """Test that chat response matches Ollama API format."""
        # Setup mock response with full Ollama response format
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "model": "phi3:mini",
            "created_at": "2026-01-25T03:05:32.87334657Z",
            "message": {"role": "assistant", "content": "Neural networks are..."},
            "done": True,
            "done_reason": "stop",
            "total_duration": 52958381914,
        }
        
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        # Call chat
        response = self.client.chat(
            model=DEFAULT_CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me about neural networks."}
            ],
            options={"temperature": 0.7}
        )
        
        # Verify response structure matches Ollama API
        self.assertIn("message", response)
        self.assertIn("model", response)
        self.assertIn("done", response)
        self.assertEqual(response["message"]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
