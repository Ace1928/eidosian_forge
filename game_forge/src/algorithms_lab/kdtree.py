"""SciPy cKDTree neighbor queries for high-performance search."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from algorithms_lab.backends import HAS_SCIPY, cKDTree
from algorithms_lab.core import Domain, WrapMode


class KDTreeNeighborSearch:
    """cKDTree-backed neighbor search with optional periodic boundaries."""

    def __init__(self, domain: Domain, leafsize: int = 32) -> None:
        if not HAS_SCIPY:
            raise ImportError("scipy is required for KDTreeNeighborSearch")
        if leafsize < 1:
            raise ValueError("leafsize must be positive")
        self.domain = domain
        self.leafsize = int(leafsize)

    def neighbor_pairs(
        self, positions: NDArray[np.float32], radius: float
    ) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Return neighbor pairs within radius using cKDTree."""

        if radius <= 0:
            raise ValueError("radius must be positive")
        pos = np.asarray(positions, dtype=np.float32)
        if self.domain.wrap == WrapMode.WRAP:
            pos = self.domain.wrap_positions(pos) - self.domain.mins
            boxsize = self.domain.sizes
        else:
            boxsize = None
        tree = cKDTree(pos, leafsize=self.leafsize, boxsize=boxsize)
        try:
            pairs = tree.query_pairs(radius, output_type="ndarray")
        except TypeError:
            pairs = np.array(list(tree.query_pairs(radius)), dtype=np.int32)
        if pairs.size == 0:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
        return pairs[:, 0].astype(np.int32), pairs[:, 1].astype(np.int32)
