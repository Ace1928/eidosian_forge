import numpy as np
import pytest

from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.kdtree import KDTreeNeighborSearch
from algorithms_lab.spatial_hash import UniformGrid

scipy = pytest.importorskip("scipy", reason="scipy required")


def test_kdtree_pairs_match_grid() -> None:
    domain = Domain(mins=np.array([0.0, 0.0]), maxs=np.array([1.0, 1.0]), wrap=WrapMode.WRAP)
    rng = np.random.default_rng(1)
    positions = rng.random((64, 2), dtype=np.float32)
    radius = 0.2
    grid = UniformGrid(domain, cell_size=radius)
    i_grid, j_grid = grid.neighbor_pairs(positions, radius=radius, backend="numpy")
    search = KDTreeNeighborSearch(domain, leafsize=16)
    i_tree, j_tree = search.neighbor_pairs(positions, radius=radius)
    pairs_grid = {tuple(sorted(pair)) for pair in zip(i_grid.tolist(), j_grid.tolist())}
    pairs_tree = {tuple(sorted(pair)) for pair in zip(i_tree.tolist(), j_tree.tolist())}
    assert pairs_grid == pairs_tree
