from __future__ import annotations

import hashlib
from typing import Any, List, Optional


class HashEmbedder:
    """Deterministic local fallback embedder with no external runtime dependency."""

    def __init__(self, dimensions: int = 16) -> None:
        self.dimensions = max(1, int(dimensions))

    def embed_text(self, text: str) -> List[float]:
        digest = hashlib.sha256(str(text).encode("utf-8", "replace")).digest()
        values = [byte / 255.0 for byte in digest[: self.dimensions]]
        if len(values) < self.dimensions:
            values.extend([0.0] * (self.dimensions - len(values)))
        return values


class ResilientEmbedder:
    """Try primary embedders first and fall back to a deterministic local embedder."""

    def __init__(self, *embedders: Any) -> None:
        self._embedders = [embedder for embedder in embedders if embedder is not None]

    def embed_text(self, text: str) -> List[float]:
        for embedder in self._embedders:
            method = getattr(embedder, "embed_text", None)
            if method is None:
                continue
            try:
                vector = method(text)
            except Exception:
                continue
            dense = _normalize_vector(vector)
            if dense:
                return dense
        return []


def build_default_embedder() -> ResilientEmbedder:
    primary: Optional[Any] = None
    try:
        from memory_forge.core.tiered_memory import OllamaEmbedder  # type: ignore

        primary = OllamaEmbedder()
    except Exception:
        primary = None
    return ResilientEmbedder(primary, HashEmbedder())


def _normalize_vector(vector: Any) -> List[float]:
    if vector is None:
        return []
    if isinstance(vector, (str, bytes)):
        return []
    try:
        values = [float(value) for value in vector]
    except Exception:
        return []
    if not values:
        return []
    return values
