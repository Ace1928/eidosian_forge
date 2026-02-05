import numpy as np

from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.spatial_hash import UniformGrid
from algorithms_lab.neighbor_list import NeighborList


def test_uniform_grid_neighbor_pairs() -> None:
    domain = Domain(mins=np.array([0.0, 0.0]), maxs=np.array([1.0, 1.0]), wrap=WrapMode.WRAP)
    positions = np.array([[0.1, 0.1], [0.2, 0.1], [0.9, 0.9]], dtype=np.float32)
    grid = UniformGrid(domain, cell_size=0.5)
    i, j = grid.neighbor_pairs(positions, radius=0.2)
    pairs = set(zip(i.tolist(), j.tolist()))
    assert (0, 1) in pairs or (1, 0) in pairs
    assert all(pair in {(0, 1), (1, 0)} for pair in pairs)


def test_neighbor_list_basic() -> None:
    domain = Domain(mins=np.array([0.0, 0.0]), maxs=np.array([1.0, 1.0]), wrap=WrapMode.WRAP)
    positions = np.array([[0.1, 0.1], [0.2, 0.1], [0.9, 0.9]], dtype=np.float32)
    nlist = NeighborList(domain, cutoff=0.2, skin=0.0)
    data = nlist.build(positions)
    neighbors = data.neighbors
    offsets = data.offsets
    idx0 = neighbors[offsets[0] : offsets[1]]
    idx1 = neighbors[offsets[1] : offsets[2]]
    assert 1 in idx0
    assert 0 in idx1
    idx2 = neighbors[offsets[2] : offsets[3]]
    assert idx2.size == 0
