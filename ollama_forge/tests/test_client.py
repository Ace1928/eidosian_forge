#!/usr/bin/env python3
"""
Tests for the OllamaClient class.

This module implements comprehensive, structurally elegant testing for the OllamaClient,
following Eidosian principles of precision, modularity, and exhaustive coverage.
"""

import json
import unittest
from typing import Any
from unittest.mock import Mock, patch

import pytest  # Add pytest import for xfail decorator

from ollama_forge.client import OllamaClient
from ollama_forge.exceptions import ModelNotFoundError, OllamaAPIError
from helpers.model_constants import (  # Updated path
        BACKUP_CHAT_MODEL,
        DEFAULT_CHAT_MODEL,
        DEFAULT_EMBEDDING_MODEL,
                )


class TestOllamaClient(unittest.TestCase):
    """Test cases for the OllamaClient class, organized by endpoint functionality."""

    def setUp(self) -> None:
        """Set up test fixtures with optimal configuration."""
        self.client = OllamaClient()

    # ===== Core API Information Tests =====

    # Update patch target to match actual implementation
    @patch("helpers.common.make_api_request")
    def test_get_version(self, mock_request: Any) -> None:
        """Test getting the Ollama version."""
        # Setup mock
        mock_response = {"version": "0.1.0"}
        mock_request.return_value = mock_response

        # Call method
        result = self.client.get_version()

        # Assert results
        mock_request.assert_called_once()
        self.assertEqual(result, {"version": "0.1.0"})

    # ===== Model Management Tests =====

    @patch("helpers.common.make_api_request")
    def test_list_models(self, mock_request: Any) -> None:
        """Test listing available models."""
        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = {"models": [{"name": "test-model"}]}
        mock_request.return_value = mock_response

        # Call method
        result = self.client.list_models()

        # Assert results
        mock_request.assert_called_once_with(
            "GET",
            "/api/tags",
            base_url=self.client.base_url,
            timeout=self.client.timeout,
        )
        self.assertEqual(result, {"models": [{"name": "test-model"}]})

    @patch("helpers.common.make_api_request")
    def test_delete_model_success(self, mock_request: Any) -> None:
        """Test deleting a model successfully."""
        # Setup mock
        mock_response = Mock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        # Call method
        result = self.client.delete_model("test-model")

        # Assert results
        mock_request.assert_called_once_with(
            "DELETE",
            "/api/delete",
            data={"model": "test-model"},
            base_url=self.client.base_url,
            timeout=self.client.timeout,
        )
        self.assertTrue(result)

    @patch("helpers.common.make_api_request")
    def test_delete_model_not_found(self, mock_request: Any) -> None:
        """Test deleting a non-existent model."""
        # Setup mock to raise the exception directly
        mock_request.side_effect = ModelNotFoundError(
            "Model 'nonexistent-model' not found"
        )

        # Assert that the correct exception is raised
        with self.assertRaises(ModelNotFoundError):
            self.client.delete_model("nonexistent-model")

    # ===== Generation Tests =====

    @patch("helpers.common.make_api_request")
    def test_generate_non_streaming(self, mock_request: Any) -> None:
        """Test generating text (non-streaming)."""
        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = {"response": "Test response"}
        mock_request.return_value = mock_response

        # Call method
        result = self.client.generate(
            DEFAULT_CHAT_MODEL, "test prompt", {"temperature": 0.7}, stream=False
        )

        # Assert results
        mock_request.assert_called_once_with(
            "POST",
            "/api/generate",
            data={
                "model": DEFAULT_CHAT_MODEL,
                "prompt": "test prompt",
                "stream": False,
                "temperature": 0.7,
            },
            base_url=self.client.base_url,
            timeout=self.client.timeout,
        )
        self.assertEqual(result, {"response": "Test response"})

    @patch("requests.post")
    def test_generate_streaming(self, mock_post: Any) -> None:
        """Test generating text (streaming)."""
        # Setup mock response with properly configured attributes
        mock_response = Mock()
        mock_response.iter_lines.return_value = [
            json.dumps({"response": "Test"}).encode(),
            json.dumps({"response": " streaming"}).encode(),
            json.dumps({"response": " response"}).encode(),
        ]
        mock_response.status_code = 200
        mock_response.text = "Test streaming response"

        # Use direct patching instead of relying on internal client methods
        with patch.object(self.client, "_with_retry", return_value=mock_response):
            # Call method
            result = self.client.generate(
                DEFAULT_CHAT_MODEL, "test prompt", stream=True
            )

            # Convert iterator to list for assertion
            chunks = list(result)

            # Assert results
            mock_post.assert_called_once()
            self.assertEqual(
                chunks,
                [
                    {"response": "Test"},
                    {"response": " streaming"},
                    {"response": " response"},
                ],
            )

    @patch("requests.post")
    def test_generate_streaming_alternative(self, mock_post: Any) -> None:
        """Test generating text (streaming) with alternative mocking approach."""
        # Setup mock response with properly configured attributes
        mock_response = Mock()
        mock_response.iter_lines.return_value = [
            json.dumps({"response": "Test"}).encode(),
            json.dumps({"response": " streaming"}).encode(),
            json.dumps({"response": " response"}).encode(),
        ]
        mock_response.status_code = 200
        mock_response.text = "Test streaming response"

        # Setup the mock correctly
        mock_post.return_value = mock_response

        # Call the method directly with more direct mocking
        with patch(
            "ollama_forge.client.requests.post", return_value=mock_response
        ) as mock_client_post:
            result = self.client.generate(
                DEFAULT_CHAT_MODEL, "test prompt", stream=True
            )

            # Convert iterator to list for assertion
            chunks = list(result)

            # Assert results
            mock_client_post.assert_called_once()
            self.assertEqual(
                chunks,
                [
                    {"response": "Test"},
                    {"response": " streaming"},
                    {"response": " response"},
                ],
            )

    # ===== Chat Tests =====

    @patch("helpers.common.make_api_request")
    def test_chat(self, mock_request: Any) -> None:
        """Test chat completion with structured messages."""
        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Hello!"}
        }
        mock_request.return_value = mock_response

        # Call method
        messages = [{"role": "user", "content": "Hi"}]
        result = self.client.chat(DEFAULT_CHAT_MODEL, messages, stream=False)

        # Assert results
        mock_request.assert_called_once_with(
            "POST",
            "/api/chat",
            data={"model": DEFAULT_CHAT_MODEL, "messages": messages, "stream": False},
            base_url=self.client.base_url,
            timeout=self.client.timeout,
        )
        self.assertEqual(result["message"]["content"], "Hello!")

    @patch("helpers.common.make_api_request")
    def test_create_embedding(self, mock_request: Any) -> None:
        """Test creating embeddings for vector representations."""
        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_request.return_value = mock_response

        # Call method
        result = self.client.create_embedding(DEFAULT_EMBEDDING_MODEL, "test text")

        # Assert results
        mock_request.assert_called_once_with(
            "POST",
            "/api/embed",
            data={"model": DEFAULT_EMBEDDING_MODEL, "prompt": "test text"},
            base_url=self.client.base_url,
            timeout=self.client.timeout,
        )
        self.assertEqual(result, {"embedding": [0.1, 0.2, 0.3]})

    # ===== Fallback & Resilience Tests =====

    @patch("helpers.common.make_api_request")
    @patch("helpers.model_constants.get_fallback_model")
    def test_generate_with_fallback(self, mock_get_fallback, mock_api_request: Any) -> None:
        """Test fallback to backup model when primary model fails."""
        # Setup fallback model
        mock_get_fallback.return_value = BACKUP_CHAT_MODEL
        
        # Setup mock to fail on primary model but succeed on backup
        def side_effect(method, endpoint, data=None, **kwargs):
            if data and data.get("model") == DEFAULT_CHAT_MODEL:
                raise OllamaAPIError("Primary model failed")
            else:
                # Return a mock object instead of a plain dict
                fallback_mock = Mock()
                fallback_mock.json.return_value = {"response": "Fallback response"}
                return fallback_mock
        
        mock_api_request.side_effect = side_effect
        
        # The client should have internal fallback logic or we'll implement it
        from ollama_forge.client import OllamaClient
        # Import the necessary function
        from helpers.model_constants import get_fallback_model

        self.client = OllamaClient()
        
        # Call method
        result = self.client.generate(
            DEFAULT_CHAT_MODEL, "test prompt", {"temperature": 0.7}, stream=False
        )
        # Assert results
        self.assertEqual(result, {"response": "Fallback response"})
        self.assertEqual(mock_api_request.call_count, 2)  # Called for both models

    # ===== Integration Tests =====
    # Note: These require an active Ollama server and are marked for conditional execution

    @pytest.mark.integration
    def test_real_get_version(self):
        """Integration test: Get actual server version."""
        try:
            version = self.client.get_version()
            self.assertIn("version", version)
        except Exception:
            self.skipTest("Ollama server not available")

    @pytest.mark.integration  
    def test_real_list_models(self):
        """Integration test: List available models from server."""
        try:
            models = self.client.list_models()
            self.assertIsInstance(models, dict)
            self.assertIn("models", models)
        except Exception:
            self.skipTest("Ollama server not available")

    @pytest.mark.integration
    def test_real_generate_text(self):
        """Integration test: Generate text with actual model."""
        try:
            response = self.client.generate(
                model=DEFAULT_CHAT_MODEL,
                prompt="Explain quantum computing in simple terms",
                options={"temperature": 0.7}
            )
            self.assertIn("response", response)
        except Exception:
            self.skipTest("Ollama server or model not available")

    @pytest.mark.integration
    def test_real_generate_text_streaming(self):
        """Integration test: Generate streaming text with actual model."""
        try:
            chunks = list(self.client.generate(
                model=DEFAULT_CHAT_MODEL,
                prompt="Write a short poem about AI",
                stream=True
            ))
            self.assertGreater(len(chunks), 0)
            self.assertIn("response", chunks[0])
        except Exception:
            self.skipTest("Ollama server or model not available")

    # ===== Error Handling Tests =====

    def test_model_not_found(self):
        """Test proper error handling when model doesn't exist."""
        with self.assertRaises(ModelNotFoundError):
            self.client.generate(model="non-existent-model", prompt="Hello")

    def test_api_error(self):
        """Test proper error handling with invalid API parameters."""
        with self.assertRaises(OllamaAPIError):
            self.client.generate(model=DEFAULT_CHAT_MODEL, prompt="", options={"temperature": -1})


if __name__ == "__main__":
    unittest.main()
