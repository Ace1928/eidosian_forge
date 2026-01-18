from falling_sand.engine.chunk import Chunk
from falling_sand.engine.materials import Material
import numpy as np
import pytest

from falling_sand.engine.simulation import (
    SimulationConfig,
    apply_boundary_transfers,
    step_chunk_lateral,
    step_chunk_vertical,
    step_world,
)


def test_step_chunk_vertical_moves_down() -> None:
    chunk = Chunk.empty((0, 0, 0), 3)
    chunk.data[1, 1, 2] = int(Material.GRANULAR)
    rng = np.random.default_rng(0)
    config = SimulationConfig(granular_spread_chance=0.0)

    updated, moved = step_chunk_vertical(chunk, rng, config)

    assert moved == 1
    assert updated[1, 1, 1] == int(Material.GRANULAR)
    assert updated[1, 1, 2] == int(Material.AIR)


def test_step_chunk_vertical_moves_up() -> None:
    chunk = Chunk.empty((0, 0, 0), 3)
    chunk.data[1, 1, 0] = int(Material.GAS)
    rng = np.random.default_rng(0)
    config = SimulationConfig()

    updated, moved = step_chunk_vertical(chunk, rng, config)

    assert moved == 1
    assert updated[1, 1, 1] == int(Material.GAS)
    assert updated[1, 1, 0] == int(Material.AIR)


def test_apply_boundary_transfers_down() -> None:
    chunk = Chunk.empty((0, 0, 0), 2)
    below = Chunk.empty((0, 0, -1), 2)
    chunk.data[0, 0, 0] = int(Material.GRANULAR)
    chunks = {(0, 0, 0): chunk, (0, 0, -1): below}

    moved = apply_boundary_transfers(chunks, (0, 0, 0))

    assert moved == 1
    assert chunk.data[0, 0, 0] == int(Material.AIR)
    assert below.data[0, 0, 1] == int(Material.GRANULAR)


def test_apply_boundary_transfers_lateral() -> None:
    chunk = Chunk.empty((0, 0, 0), 2)
    right = Chunk.empty((1, 0, 0), 2)
    chunk.data[1, 0, 0] = int(Material.LIQUID)
    chunks = {(0, 0, 0): chunk, (1, 0, 0): right}

    moved = apply_boundary_transfers(chunks, (0, 0, 0))

    assert moved == 1
    assert right.data[0, 0, 0] == int(Material.LIQUID)


def test_step_world_updates_dirty() -> None:
    chunk = Chunk.empty((0, 0, 0), 2)
    chunk.data[0, 0, 1] = int(Material.GRANULAR)
    chunks = {(0, 0, 0): chunk}

    stats = step_world(chunks)

    assert stats.moved_voxels >= 1
    assert chunk.dirty


def test_step_chunk_lateral_moves_liquid() -> None:
    chunk = Chunk.empty((0, 0, 0), 3)
    chunk.data[1, 1, 1] = int(Material.LIQUID)
    rng = np.random.default_rng(0)
    config = SimulationConfig(liquid_spread_chance=1.0)

    updated, moved = step_chunk_lateral(chunk, rng, config)

    assert moved >= 1
    assert int(Material.LIQUID) in updated


def test_simulation_config_validation() -> None:
    with pytest.raises(ValueError, match="granular_spread_chance must be between 0 and 1"):
        SimulationConfig(granular_spread_chance=1.5)
