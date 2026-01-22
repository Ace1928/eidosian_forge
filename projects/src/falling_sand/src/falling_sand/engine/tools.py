"""Voxel tool operations for placing and removing materials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from falling_sand.engine.materials import Material
from falling_sand.engine.world import World


VoxelCoord = Tuple[int, int, int]


@dataclass(frozen=True)
class ToolConfig:
    """Configuration for voxel tools."""

    radius: int = 1

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("radius must be positive")


def _iter_sphere(center: VoxelCoord, radius: int) -> Iterable[VoxelCoord]:
    cx, cy, cz = center
    r2 = radius * radius
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx * dx + dy * dy + dz * dz <= r2:
                    yield (cx + dx, cy + dy, cz + dz)


def place_sphere(world: World, center: VoxelCoord, material: Material, radius: int) -> int:
    """Place material in a spherical radius."""

    if radius <= 0:
        raise ValueError("radius must be positive")
    count = 0
    for voxel in _iter_sphere(center, radius):
        world.set_voxel(voxel, material)
        count += 1
    return count


def erase_sphere(world: World, center: VoxelCoord, radius: int) -> int:
    """Erase materials in a spherical radius."""

    if radius <= 0:
        raise ValueError("radius must be positive")
    count = 0
    for voxel in _iter_sphere(center, radius):
        world.set_voxel(voxel, Material.AIR)
        count += 1
    return count
