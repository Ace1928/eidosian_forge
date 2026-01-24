"""Embedding helper utilities for ollama_forge."""

import httpx
from typing import List, Optional


def create_embedding(
    text: str,
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
) -> Optional[List[float]]:
    """Create an embedding for a single text."""
    try:
        with httpx.Client() as client:
            resp = client.post(
                f"{base_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=30.0
            )
            if resp.status_code == 200:
                return resp.json().get("embedding", [])
    except Exception:
        pass
    return None


def create_embeddings(
    texts: List[str],
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
) -> List[Optional[List[float]]]:
    """Create embeddings for multiple texts."""
    return [create_embedding(text, model, base_url) for text in texts]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


# Alias for compatibility
calculate_similarity = cosine_similarity


def normalize_vector(v: List[float]) -> List[float]:
    """Normalize a vector to unit length."""
    if not v:
        return []
    norm = sum(x * x for x in v) ** 0.5
    if norm == 0:
        return v
    return [x / norm for x in v]


def batch_calculate_similarities(
    query: List[float],
    candidates: List[List[float]],
) -> List[float]:
    """Calculate similarities between a query and multiple candidates."""
    return [cosine_similarity(query, c) for c in candidates]
