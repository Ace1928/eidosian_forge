#!/usr/bin/env python3
"""
Tests for the OllamaClient class.

Tests properly mock httpx to test the client's API interactions.
"""

import unittest
from typing import Any
from unittest.mock import Mock, patch, MagicMock

import pytest

from ollama_forge.client import OllamaClient
from ollama_forge.exceptions import ModelNotFoundError, OllamaAPIError
from helpers.model_constants import DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL


class TestOllamaClient(unittest.TestCase):
    """Test cases for the OllamaClient class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.client = OllamaClient()

    # ===== Core API Information Tests =====

    @patch("ollama_forge.client.httpx.Client")
    def test_get_version(self, mock_client_class: Any) -> None:
        """Test getting the Ollama version."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"version": "0.1.0"}
        
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        result = self.client.get_version()

        mock_client.get.assert_called_once()
        self.assertEqual(result, {"version": "0.1.0"})

    # ===== Model Management Tests =====

    @patch("ollama_forge.client.httpx.Client")
    def test_list_models(self, mock_client_class: Any) -> None:
        """Test listing available models."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "test-model"}]}
        
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        result = self.client.list_models()

        mock_client.get.assert_called_once()
        self.assertEqual(result, ["test-model"])

    @patch("ollama_forge.client.httpx.Client")
    def test_delete_model_success(self, mock_client_class: Any) -> None:
        """Test deleting a model successfully."""
        mock_response = Mock()
        mock_response.status_code = 200
        
        mock_client = MagicMock()
        mock_client.request.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        result = self.client.delete_model("test-model")

        self.assertTrue(result)

    @patch("ollama_forge.client.httpx.Client")
    def test_delete_model_not_found(self, mock_client_class: Any) -> None:
        """Test deleting a non-existent model."""
        mock_response = Mock()
        mock_response.status_code = 404
        
        mock_client = MagicMock()
        mock_client.request.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        result = self.client.delete_model("nonexistent-model")

        self.assertFalse(result)

    # ===== Generate Tests =====

    @patch("ollama_forge.client.httpx.Client")
    def test_generate_non_streaming(self, mock_client_class: Any) -> None:
        """Test non-streaming text generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "model": "test-model",
            "response": "Generated text",
            "done": True
        }
        
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        result = self.client.generate(
            model="test-model",
            prompt="Hello"
        )

        mock_client.post.assert_called_once()
        self.assertEqual(result.response, "Generated text")
        self.assertTrue(result.done)

    @patch("ollama_forge.client.httpx.Client")
    def test_generate_streaming_not_implemented(self, mock_client_class: Any) -> None:
        """Test that streaming generation raises NotImplementedError."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        with self.assertRaises(NotImplementedError):
            self.client.generate(
                model="test-model",
                prompt="Hello",
                stream=True
            )

    # ===== Chat Tests =====

    @patch("ollama_forge.client.httpx.Client")
    def test_chat(self, mock_client_class: Any) -> None:
        """Test chat functionality."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "model": DEFAULT_CHAT_MODEL,
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True
        }
        
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        result = self.client.chat(
            model=DEFAULT_CHAT_MODEL,
            messages=[{"role": "user", "content": "Hi"}]
        )

        mock_client.post.assert_called_once()
        self.assertEqual(result["message"]["content"], "Hello!")

    # ===== Embedding Tests =====

    @patch("ollama_forge.client.httpx.Client")
    def test_create_embedding(self, mock_client_class: Any) -> None:
        """Test embedding creation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "model": DEFAULT_EMBEDDING_MODEL,
            "embeddings": [[0.1, 0.2, 0.3]]
        }
        
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        result = self.client.create_embedding(
            model=DEFAULT_EMBEDDING_MODEL,
            prompt="Test text"
        )

        mock_client.post.assert_called_once()
        self.assertIn("embeddings", result)

    # ===== Error Handling Tests =====

    @patch("ollama_forge.client.httpx.Client")
    def test_api_error(self, mock_client_class: Any) -> None:
        """Test API error handling."""
        import httpx
        
        mock_request = Mock()
        mock_request.url = "http://localhost:11434/api/generate"
        
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.reason_phrase = "Internal Server Error"
        mock_response.is_success = False
        mock_response.has_redirect_location = False
        mock_response._request = mock_request
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error",
            request=mock_request,
            response=mock_response
        )
        
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        with self.assertRaises(httpx.HTTPStatusError):
            self.client.generate(model="test", prompt="hello")

    # ===== Model Information Tests =====

    @patch("ollama_forge.client.httpx.Client")
    def test_show_model(self, mock_client_class: Any) -> None:
        """Test showing model information."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "modelfile": "# Modelfile",
            "parameters": "temperature 0.7",
            "template": "{{ .Prompt }}"
        }
        
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        result = self.client.show_model("test-model")

        mock_client.post.assert_called_once()
        self.assertIn("modelfile", result)

    @patch("ollama_forge.client.httpx.Client")
    def test_pull_model(self, mock_client_class: Any) -> None:
        """Test pulling a model."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"status": "success"}
        
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        result = self.client.pull_model("test-model")

        mock_client.post.assert_called_once()
        self.assertEqual(result["status"], "success")


class TestOllamaClientIntegration(unittest.TestCase):
    """Integration tests that require a running Ollama server."""
    
    @pytest.mark.integration
    def test_real_list_models(self):
        """Test listing models against real server."""
        client = OllamaClient()
        try:
            models = client.list_models()
            self.assertIsInstance(models, list)
        except Exception:
            pytest.skip("Ollama server not available")

    @pytest.mark.integration
    def test_real_get_version(self):
        """Test getting version from real server."""
        client = OllamaClient()
        try:
            version = client.get_version()
            self.assertIn("version", version)
        except Exception:
            pytest.skip("Ollama server not available")


if __name__ == "__main__":
    unittest.main()
