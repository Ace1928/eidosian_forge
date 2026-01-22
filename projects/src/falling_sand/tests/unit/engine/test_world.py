import numpy as np

from falling_sand.engine.config import VoxelConfig
from falling_sand.engine.materials import Material
from falling_sand.engine.world import World


def test_world_set_get_voxel() -> None:
    world = World(config=VoxelConfig())
    world.set_voxel((0, 0, 0), Material.GRANULAR)
    assert world.get_voxel((0, 0, 0)) == Material.GRANULAR
    assert world.get_voxel((99, 99, 99)) == Material.AIR


def test_world_fill_and_prune() -> None:
    world = World(config=VoxelConfig())
    world.fill_chunk((0, 0, 0), Material.AIR)
    assert world.get_chunk((0, 0, 0)) is not None
    world.prune_empty()
    assert world.get_chunk((0, 0, 0)) is None


def test_world_snapshot() -> None:
    world = World(config=VoxelConfig())
    world.set_voxel((1, 2, 3), Material.SOLID)
    snap = world.snapshot()
    assert (0, 0, 0) in snap
    assert isinstance(snap[(0, 0, 0)], np.ndarray)


def test_surface_height() -> None:
    world = World(config=VoxelConfig())
    world.set_voxel((2, 2, 3), Material.SOLID)
    assert world.surface_height(2, 2, default=0) == 3
    assert world.surface_height(9, 9, default=1) == 1
