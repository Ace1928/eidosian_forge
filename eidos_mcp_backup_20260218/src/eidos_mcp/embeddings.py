from __future__ import annotations

import hashlib
from typing import List
from eidosian_core import eidosian


class SimpleEmbedder:
    """Deterministic lightweight embedder for local similarity scoring."""

    def __init__(self, dimensions: int = 16) -> None:
        self.dimensions = max(1, dimensions)

    @eidosian()
    def embed_text(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [b / 255.0 for b in digest[: self.dimensions]]
        if len(values) < self.dimensions:
            values.extend([0.0] * (self.dimensions - len(values)))
        return values
