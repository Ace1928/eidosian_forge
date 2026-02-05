import numpy as np
import pytest

from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.spatial_hash import UniformGrid

numba = pytest.importorskip("numba", reason="numba required")


def test_numba_backend_matches_numpy() -> None:
    domain = Domain(mins=np.array([0.0, 0.0]), maxs=np.array([1.0, 1.0]), wrap=WrapMode.WRAP)
    rng = np.random.default_rng(0)
    positions = rng.random((64, 2), dtype=np.float32)
    grid = UniformGrid(domain, cell_size=0.2)
    i_np, j_np = grid.neighbor_pairs(positions, radius=0.2, backend="numpy")
    i_nb, j_nb = grid.neighbor_pairs(positions, radius=0.2, backend="numba")
    pairs_np = set(zip(i_np.tolist(), j_np.tolist()))
    pairs_nb = set(zip(i_nb.tolist(), j_nb.tolist()))
    assert pairs_np == pairs_nb
