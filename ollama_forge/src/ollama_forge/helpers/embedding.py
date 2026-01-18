#!/usr/bin/env python3
"""
Embedding utilities for Ollama Forge.

This module provides functions for working with vector embeddings,
including similarity calculations, normalization, and batch processing.
Following Eidosian principles of exhaustive but concise implementation.

Handles vector operations with NumPy acceleration when available,
with automatic fallback to pure Python implementations.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
from numpy.typing import NDArray

# Type definitions
VectorType = List[float]
MatrixType = List[VectorType]
EmbeddingIndex = int
SimilarityScore = float
SimilarityPair = Tuple[EmbeddingIndex, SimilarityScore]
ResponseType = Dict[
    str, Union[List[float], List[List[float]], Dict[str, Any], List[Any]]
]
T = TypeVar("T")  # Generic type for matrix operations

# Initialize logger
logger = logging.getLogger(__name__)

# Runtime NumPy availability check
# Define with initial value
numpy_available = False
try:
    np.array([1.0]).dot(np.array([1.0]))
    numpy_available = True
except (ImportError, AttributeError):
    logger.warning("NumPy unavailable: falling back to pure Python implementations")

# Constant reference to runtime check result
NUMPY_AVAILABLE = numpy_available


def calculate_similarity(vec1: VectorType, vec2: VectorType) -> float:
    """
    Calculate cosine similarity between two vectors.

    Measures the cosine of the angle between two vectors,
    providing a similarity score between -1 and 1.

    Args:
        vec1: First vector as a list of floats
        vec2: Second vector as a list of floats

    Returns:
        float: Cosine similarity value between -1 and 1

    Raises:
        ValueError: If vectors are empty or have different dimensions

    Examples:
        >>> calculate_similarity([1.0, 0.0], [0.0, 1.0])
        0.0
        >>> calculate_similarity([1.0, 1.0], [1.0, 1.0])
        1.0
    """
    if not vec1 or not vec2:
        raise ValueError("Vectors cannot be empty")

    if len(vec1) != len(vec2):
        raise ValueError(f"Vector dimensions don't match: {len(vec1)} vs {len(vec2)}")

    if NUMPY_AVAILABLE:
        # NumPy implementation for performance
        v1 = np.array(vec1, dtype=np.float64)
        v2 = np.array(vec2, dtype=np.float64)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        # Handle zero vectors gracefully
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))
    else:
        # Pure Python implementation as fallback
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        # Handle zero vectors gracefully
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        return dot_product / (norm1 * norm2)


def normalize_vector(vector: VectorType) -> VectorType:
    """
    Normalize a vector to unit length (magnitude of 1).

    Scales a vector to have a Euclidean norm (magnitude) of 1,
    preserving its direction while standardizing its length.

    Args:
        vector: Input vector as a list of floats

    Returns:
        List[float]: Normalized vector with unit length

    Raises:
        ValueError: If vector is empty

    Examples:
        >>> normalize_vector([3.0, 4.0])  # 3-4-5 triangle
        [0.6, 0.8]
    """
    if not vector:
        raise ValueError("Vector cannot be empty")

    if NUMPY_AVAILABLE:
        # NumPy implementation
        v = np.array(vector, dtype=np.float64)
        norm = np.linalg.norm(v)

        # Handle zero vector gracefully
        if norm < 1e-10:
            logger.warning("Cannot normalize zero vector, returning original")
            return vector

        return cast(VectorType, (v / norm).tolist())
    else:
        # Pure Python implementation
        norm = math.sqrt(sum(x * x for x in vector))

        # Handle zero vector gracefully
        if norm < 1e-10:
            logger.warning("Cannot normalize zero vector, returning original")
            return vector

        # Return normalized vector
        return [x / norm for x in vector]


def batch_calculate_similarities(
    query_vector: VectorType, comparison_vectors: MatrixType
) -> List[SimilarityPair]:
    """
    Calculate similarities between a query vector and multiple comparison vectors.

    Efficiently computes cosine similarity between a single query vector
    and a batch of comparison vectors, enabling vector search operations.

    Args:
        query_vector: Query vector to compare against
        comparison_vectors: List of vectors to compare with query

    Returns:
        List[Tuple[int, float]]: List of (index, similarity) pairs

    Examples:
        >>> batch_calculate_similarities([1.0, 1.0], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        [(2, 1.0), (0, 0.7071...), (1, 0.7071...)]
    """
    if not comparison_vectors:
        return []

    if NUMPY_AVAILABLE:
        # Vectorized implementation for large batches
        query_array = np.array(query_vector, dtype=np.float64)
        comparison_array = np.array(comparison_vectors, dtype=np.float64)

        # Normalize the query vector
        query_norm = np.linalg.norm(query_array)
        if query_norm > 0:
            query_array = query_array / query_norm

        # Calculate all similarities at once
        norms = np.linalg.norm(comparison_array, axis=1)
        valid_indices = norms > 0

        # Initialize similarities array
        similarities_array = np.zeros(len(comparison_vectors))
        if any(valid_indices):
            normalized_vectors = (
                comparison_array[valid_indices] / norms[valid_indices, np.newaxis]
            )
            similarities_array[valid_indices] = np.dot(normalized_vectors, query_array)

        # Convert to the required return type
        return [(int(i), float(sim)) for i, sim in enumerate(similarities_array)]
    else:
        # Fallback implementation
        similarities: List[SimilarityPair] = []
        for i, vec in enumerate(comparison_vectors):
            try:
                sim = calculate_similarity(query_vector, vec)
                similarities.append((i, sim))
            except ValueError as e:
                logger.warning(f"Skipping vector {i}: {e}")
                similarities.append((i, 0.0))

        return similarities


def process_embeddings_response(response: ResponseType) -> Optional[VectorType]:
    """
    Process the response from an embedding API call to extract the embedding vector.

    Handles different response formats from various embedding API providers
    and versions, extracting the actual vector representation.

    Args:
        response: API response containing embeddings in various formats

    Returns:
        Optional[List[float]]: Extracted embedding vector or None if extraction fails

    Examples:
        >>> process_embeddings_response({"embedding": [0.1, 0.2, 0.3]})
        [0.1, 0.2, 0.3]
        >>> process_embeddings_response({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
        [0.1, 0.2]
    """
    if not response:
        logger.error("Empty response received")
        return None

    # Handle single embedding format
    if "embedding" in response:
        embedding_data = response["embedding"]
        if isinstance(embedding_data, list):
            # Ensure it's a flat list of numbers
            if all(isinstance(x, (int, float)) for x in embedding_data):
                return cast(VectorType, embedding_data)

    # Handle multiple embeddings format
    elif "embeddings" in response:
        embeddings_data = response["embeddings"]

        # Case: Direct list of floats
        if isinstance(embeddings_data, list):
            if embeddings_data and all(
                isinstance(x, (int, float)) for x in embeddings_data
            ):
                return cast(VectorType, embeddings_data)

            # Case: List of embedding vectors (take first)
            if embeddings_data and isinstance(embeddings_data[0], list):
                # Type check and convert
                first_embedding = cast(List[Union[int, float]], embeddings_data[0])
                for x in first_embedding:
                    return cast(VectorType, first_embedding)

    # Log what keys were found in the response for debugging
    key_list = list(str(k) for k in response.keys())
    logger.error(f"Could not extract embedding from response: {key_list}")
    return None


def create_embedding_matrix(
    embeddings: MatrixType, normalize: bool = True
) -> Union[MatrixType, NDArray[np.float64]]:
    """
    Convert a list of embeddings into a matrix for efficient operations.

    Creates a matrix representation of multiple embedding vectors,
    optionally normalizing them for cosine similarity operations.

    Args:
        embeddings: List of embedding vectors
        normalize: Whether to normalize the vectors to unit length

    Returns:
        Union[List[List[float]], NDArray]: Matrix of embeddings
        (numpy array if available, else nested list)

    Examples:
        >>> create_embedding_matrix([[1.0, 1.0], [1.0, 0.0]])
        array([[0.7071..., 0.7071...],
               [1.0, 0.0]])
    """
    if not embeddings:
        return []

    # Normalize if requested
    processed_embeddings = embeddings
    if normalize:
        processed_embeddings = [normalize_vector(vec) for vec in embeddings]

    # Return as NumPy array if available, otherwise as list
    if NUMPY_AVAILABLE:
        return np.array(processed_embeddings, dtype=np.float64)
    else:
        return processed_embeddings


def top_k_similarities(
    query_embedding: VectorType,
    embedding_matrix: Union[MatrixType, NDArray[np.float64]],
    k: int = 5,
) -> List[SimilarityPair]:
    """
    Find top k most similar vectors to a query embedding.

    Efficiently identifies the k most similar vectors to a query,
    returning them sorted by similarity score in descending order.

    Args:
        query_embedding: Query vector to compare against
        embedding_matrix: Matrix of vectors to search through
        k: Number of top matches to return (default: 5)

    Returns:
        List[Tuple[int, float]]: List of (index, similarity) pairs for top k matches,
        sorted by descending similarity

    Examples:
        >>> top_k_similarities([1.0, 0.0], [[0.0, 1.0], [0.7, 0.7], [1.0, 0.0]], k=2)
        [(2, 1.0), (1, 0.7071...)]
    """
    # Validate inputs
    if not embedding_matrix:
        return []

    if k <= 0:
        return []

    # Ensure k doesn't exceed matrix size
    matrix_size = (
        len(embedding_matrix)
        if isinstance(embedding_matrix, list)
        else embedding_matrix.shape[0]
    )
    k = min(k, matrix_size)

    # Use vectorized NumPy implementation when available
    if NUMPY_AVAILABLE and isinstance(embedding_matrix, np.ndarray):
        # Normalize query vector
        query_array = np.array(query_embedding, dtype=np.float64)
        query_norm = np.linalg.norm(query_array)

        if query_norm > 0:
            query_norm_vector = query_array / query_norm

            # Calculate dot products in one vectorized operation
            similarities = np.dot(embedding_matrix, query_norm_vector)

            # Get top k indices
            top_indices = np.argsort(-similarities)[:k]
            return [(int(i), float(similarities[i])) for i in top_indices]
        else:
            # Zero query vector case - all similarities are 0
            return [(i, 0.0) for i in range(min(k, matrix_size))]
    else:
        # Fallback to pure Python implementation
        all_similarities = batch_calculate_similarities(
            query_embedding, cast(MatrixType, embedding_matrix)
        )
        sorted_similarities = sorted(all_similarities, key=lambda x: x[1], reverse=True)
        return sorted_similarities[:k]
