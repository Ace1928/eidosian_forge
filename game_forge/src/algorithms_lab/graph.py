"""Neighbor graph construction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from algorithms_lab.backends import HAS_SCIPY
from algorithms_lab.core import Domain, WrapMode, ensure_f32
from algorithms_lab.neighbors import NeighborSearch
from algorithms_lab.spatial_hash import UniformGrid


@dataclass(frozen=True)
class NeighborGraph:
    """Directed neighbor graph in COO form."""

    rows: NDArray[np.int32]
    cols: NDArray[np.int32]
    dist: NDArray[np.float32]


def build_neighbor_graph(
    positions: NDArray[np.float32],
    radius: float,
    domain: Domain,
    method: str = "auto",
    backend: str = "auto",
    leafsize: int = 32,
) -> NeighborGraph:
    """Build a directed neighbor graph (rows, cols, dist).

    The graph is directed: for each undirected neighbor pair (i, j), the
    output includes both (i, j) and (j, i).
    """

    if radius <= 0:
        return NeighborGraph(
            rows=np.zeros(0, dtype=np.int32),
            cols=np.zeros(0, dtype=np.int32),
            dist=np.zeros(0, dtype=np.float32),
        )
    pos = ensure_f32(positions)
    if method not in ("auto", "grid", "kdtree"):
        raise ValueError("method must be one of: auto, grid, kdtree")
    if method == "kdtree" and not HAS_SCIPY:
        raise ImportError("scipy is required for method='kdtree'")

    if method == "grid" or (method == "auto" and not HAS_SCIPY):
        grid = UniformGrid(domain, cell_size=radius)
        i, j = grid.neighbor_pairs(pos, radius=radius, backend=backend)
    else:
        search = NeighborSearch(domain, radius=radius, method=method, backend=backend, leafsize=leafsize)
        i, j = search.pairs(pos)

    if i.size == 0:
        return NeighborGraph(
            rows=np.zeros(0, dtype=np.int32),
            cols=np.zeros(0, dtype=np.int32),
            dist=np.zeros(0, dtype=np.float32),
        )

    # Directed edges.
    rows = np.concatenate([i, j]).astype(np.int32)
    cols = np.concatenate([j, i]).astype(np.int32)
    delta = pos[rows] - pos[cols]
    delta = domain.minimal_image(delta)
    dist = np.linalg.norm(delta, axis=1).astype(np.float32)
    mask = dist > 0.0
    return NeighborGraph(rows=rows[mask], cols=cols[mask], dist=dist[mask])
