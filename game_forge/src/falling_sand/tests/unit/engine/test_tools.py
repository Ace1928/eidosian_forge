import pytest

from falling_sand.engine.config import VoxelConfig
from falling_sand.engine.materials import Material
from falling_sand.engine.tools import erase_sphere, place_sphere
from falling_sand.engine.world import World


def test_place_sphere() -> None:
    world = World(config=VoxelConfig())
    placed = place_sphere(world, (0, 0, 0), Material.SOLID, radius=1)
    assert placed > 0
    assert world.get_voxel((0, 0, 0)) == Material.SOLID


def test_erase_sphere() -> None:
    world = World(config=VoxelConfig())
    place_sphere(world, (0, 0, 0), Material.GRANULAR, radius=1)
    erased = erase_sphere(world, (0, 0, 0), radius=1)
    assert erased > 0
    assert world.get_voxel((0, 0, 0)) == Material.AIR


def test_tool_validation() -> None:
    world = World(config=VoxelConfig())
    with pytest.raises(ValueError, match="radius must be positive"):
        place_sphere(world, (0, 0, 0), Material.SOLID, radius=0)
    with pytest.raises(ValueError, match="radius must be positive"):
        erase_sphere(world, (0, 0, 0), radius=0)
