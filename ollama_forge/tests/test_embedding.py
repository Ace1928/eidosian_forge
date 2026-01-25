#!/usr/bin/env python3
"""
Tests for the embedding functionality.
"""
import unittest
from typing import Any
from unittest.mock import Mock, patch, MagicMock

from helpers.model_constants import DEFAULT_EMBEDDING_MODEL
from ollama_forge import OllamaClient


class TestEmbeddings(unittest.TestCase):
    """Test cases for OllamaClient embedding functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.client = OllamaClient()

    @patch("ollama_forge.client.httpx.Client")
    def test_create_embedding(self, mock_client_class: Any) -> None:
        """Test creating embeddings with the primary model."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "model": DEFAULT_EMBEDDING_MODEL,
            "embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5]]
        }
        
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        result = self.client.create_embedding(
            model=DEFAULT_EMBEDDING_MODEL,
            prompt="Test text for embedding"
        )

        mock_client.post.assert_called_once()
        self.assertIn("embeddings", result)
        self.assertEqual(len(result["embeddings"][0]), 5)

    @patch("ollama_forge.client.httpx.Client")
    def test_create_embedding_error(self, mock_client_class: Any) -> None:
        """Test embedding creation error handling."""
        import httpx
        
        mock_request = Mock()
        mock_request.url = "http://localhost:11434/api/embed"
        
        mock_response = Mock()
        mock_response.status_code = 500
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
            self.client.create_embedding(
                model=DEFAULT_EMBEDDING_MODEL,
                prompt="Test text"
            )

    @patch("ollama_forge.client.httpx.Client")
    def test_batch_embeddings(self, mock_client_class: Any) -> None:
        """Test batch embedding creation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        # Return different embeddings for each call
        mock_response.json.side_effect = [
            {"embeddings": [[0.1, 0.2, 0.3]]},
            {"embeddings": [[0.4, 0.5, 0.6]]},
            {"embeddings": [[0.7, 0.8, 0.9]]},
        ]
        
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        results = self.client.batch_embeddings(
            model="nomic-embed-text",
            prompts=["Text one", "Text two", "Text three"]
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(mock_client.post.call_count, 3)


class TestEmbeddingSimilarity(unittest.TestCase):
    """Test cases for embedding similarity calculations."""

    def test_calculate_similarity_identical(self) -> None:
        """Test similarity of identical vectors."""
        from helpers.embedding import calculate_similarity
        
        vec = [0.1, 0.2, 0.3, 0.4, 0.5]
        similarity = calculate_similarity(vec, vec)
        
        self.assertAlmostEqual(similarity, 1.0, places=5)

    def test_calculate_similarity_orthogonal(self) -> None:
        """Test similarity of orthogonal vectors."""
        from helpers.embedding import calculate_similarity
        
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        similarity = calculate_similarity(vec1, vec2)
        
        self.assertAlmostEqual(similarity, 0.0, places=5)

    def test_calculate_similarity_opposite(self) -> None:
        """Test similarity of opposite vectors."""
        from helpers.embedding import calculate_similarity
        
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = calculate_similarity(vec1, vec2)
        
        self.assertAlmostEqual(similarity, -1.0, places=5)


if __name__ == "__main__":
    unittest.main()
