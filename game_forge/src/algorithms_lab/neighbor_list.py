"""Verlet-style neighbor list built on a uniform grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from algorithms_lab.core import Domain, ensure_f32, ensure_i32
from algorithms_lab.spatial_hash import UniformGrid


@dataclass(frozen=True)
class NeighborListData:
    """Compressed neighbor list (CSR format)."""

    neighbors: NDArray[np.int32]
    offsets: NDArray[np.int32]


class NeighborList:
    """Maintain a Verlet neighbor list with optional skin distance.

    The neighbor enumeration backend can be set to `numpy` or `numba` for
    accelerated pair generation.
    """

    def __init__(
        self,
        domain: Domain,
        cutoff: float,
        skin: float = 0.0,
        backend: str = "auto",
    ) -> None:
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        if skin < 0:
            raise ValueError("skin must be non-negative")
        if backend not in ("auto", "numpy", "numba"):
            raise ValueError("backend must be one of: auto, numpy, numba")
        self.domain = domain
        self.cutoff = float(cutoff)
        self.skin = float(skin)
        self.radius = self.cutoff + self.skin
        self.backend = backend
        self._grid = UniformGrid(domain, cell_size=self.radius)
        self._last_positions: NDArray[np.float32] | None = None
        self._data: NeighborListData | None = None

    def needs_rebuild(self, positions: NDArray[np.float32]) -> bool:
        """Return True if any particle moved more than half the skin distance."""

        if self._last_positions is None:
            return True
        pos = ensure_f32(positions)
        delta = pos - self._last_positions
        delta = self.domain.minimal_image(delta)
        disp2 = np.einsum("ij,ij->i", delta, delta)
        threshold = (self.skin * 0.5) ** 2
        return bool(np.any(disp2 > threshold))

    def build(self, positions: NDArray[np.float32]) -> NeighborListData:
        """Build and store the neighbor list for positions."""

        pos = ensure_f32(positions)
        pair_i, pair_j = self._grid.neighbor_pairs(
            pos, radius=self.radius, backend=self.backend
        )
        if pair_i.size == 0:
            neighbors = np.zeros(0, dtype=np.int32)
            offsets = np.zeros(pos.shape[0] + 1, dtype=np.int32)
            data = NeighborListData(neighbors=neighbors, offsets=offsets)
            self._data = data
            self._last_positions = pos.copy()
            return data
        # Build symmetric neighbor list.
        orig_i = pair_i
        orig_j = pair_j
        pair_i = np.concatenate([orig_i, orig_j])
        pair_j = np.concatenate([orig_j, orig_i])
        order = np.argsort(pair_i, kind="stable")
        pair_i = pair_i[order]
        pair_j = pair_j[order]
        counts = np.bincount(pair_i, minlength=pos.shape[0]).astype(np.int32)
        offsets = np.concatenate(
            [np.array([0], dtype=np.int32), np.cumsum(counts, dtype=np.int32)]
        )
        data = NeighborListData(neighbors=ensure_i32(pair_j), offsets=ensure_i32(offsets))
        self._data = data
        self._last_positions = pos.copy()
        return data

    def get(self, positions: NDArray[np.float32]) -> NeighborListData:
        """Return the current neighbor list, rebuilding if needed."""

        if self._data is None or self.needs_rebuild(positions):
            return self.build(positions)
        return self._data
