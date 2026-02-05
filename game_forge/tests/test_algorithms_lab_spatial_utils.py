import numpy as np
import pytest

from algorithms_lab.backends import HAS_NUMBA
from algorithms_lab.spatial_utils import (
    GridConfig,
    adaptive_cell_size,
    compute_batch_ranges,
    compute_cell_densities,
    compute_morton_order,
    morton_decode_2d,
    morton_encode_2d,
    pack_positions_soa,
    prefetch_neighbor_data,
)


def test_grid_config_from_world():
    cfg = GridConfig.from_world(world_size=10.0, interaction_radius=0.5, n_particles=1000)
    assert cfg.cell_size >= 0.5
    assert cfg.grid_width > 0
    assert cfg.grid_height > 0
    assert cfg.max_per_cell > 0


def test_morton_round_trip():
    for x in (0, 1, 7, 255):
        for y in (0, 3, 12, 255):
            code = morton_encode_2d(x, y)
            dx, dy = morton_decode_2d(code)
            assert dx == x
            assert dy == y


def test_compute_morton_order():
    pos = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]], dtype=np.float32)
    order = compute_morton_order(pos, pos.shape[0], cell_size=0.5, half_world=5.0, grid_w=20, grid_h=20)
    assert order.shape[0] == pos.shape[0]
    assert set(order.tolist()) == {0, 1, 2}


def test_compute_cell_densities():
    grid_counts = np.array([[0, 2], [3, 0]], dtype=np.int32)
    avg, max_d, empty = compute_cell_densities(grid_counts)
    assert max_d == 3.0
    assert empty == 2
    assert avg == pytest.approx(5.0 / 4.0, rel=1e-3)


def test_adaptive_cell_size():
    assert adaptive_cell_size(50.0, 200.0, 1.0) > 1.0
    assert adaptive_cell_size(1.0, 2.0, 1.0) < 1.0


def test_compute_batch_ranges():
    ranges = compute_batch_ranges(100, batch_size=32)
    assert ranges.shape[1] == 2
    assert ranges[0, 0] == 0
    assert ranges[-1, 1] == 100


@pytest.mark.skipif(not HAS_NUMBA, reason="numba required for SoA helpers")
def test_pack_positions_soa():
    pos = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x_arr, y_arr = pack_positions_soa(pos, pos.shape[0])
    assert np.allclose(x_arr, np.array([1.0, 3.0], dtype=np.float32))
    assert np.allclose(y_arr, np.array([2.0, 4.0], dtype=np.float32))


@pytest.mark.skipif(not HAS_NUMBA, reason="numba required for prefetch helper")
def test_prefetch_neighbor_data():
    pos = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    colors = np.array([1, 2], dtype=np.int32)
    angle = np.array([0.0, 0.5], dtype=np.float32)
    cell_particles = np.array([1, 0], dtype=np.int32)
    n_pos, n_colors, n_angles = prefetch_neighbor_data(
        pos, colors, angle, cell_particles, cell_count=2, max_neighbors=2
    )
    assert n_pos.shape == (2, 2)
    assert n_colors.tolist() == [2, 1]
    assert n_angles.tolist() == [0.5, 0.0]
