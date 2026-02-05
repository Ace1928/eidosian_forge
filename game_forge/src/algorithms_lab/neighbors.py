"""Unified neighbor search interfaces with optional backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from algorithms_lab.backends import HAS_SCIPY
from algorithms_lab.core import Domain
from algorithms_lab.kdtree import KDTreeNeighborSearch
from algorithms_lab.spatial_hash import UniformGrid


@dataclass
class NeighborSearch:
    """Neighbor search abstraction for algorithms_lab modules."""

    domain: Domain
    radius: float
    method: str = "auto"
    backend: str = "auto"
    leafsize: int = 32

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if self.method not in ("auto", "grid", "kdtree"):
            raise ValueError("method must be one of: auto, grid, kdtree")
        if self.backend not in ("auto", "numpy", "numba"):
            raise ValueError("backend must be one of: auto, numpy, numba")
        if self.method == "kdtree" and not HAS_SCIPY:
            raise ImportError("scipy is required for method='kdtree'")
        if self.leafsize < 1:
            raise ValueError("leafsize must be positive")
        self._grid = UniformGrid(self.domain, cell_size=self.radius)
        self._kdtree = None
        if self.method in ("kdtree", "auto") and HAS_SCIPY:
            self._kdtree = KDTreeNeighborSearch(self.domain, leafsize=self.leafsize)

    def pairs(self, positions: NDArray[np.float32]) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Return neighbor pairs using the configured search method."""

        if self._kdtree is not None and self.method in ("kdtree", "auto"):
            return self._kdtree.neighbor_pairs(positions, radius=self.radius)
        return self._grid.neighbor_pairs(positions, radius=self.radius, backend=self.backend)
