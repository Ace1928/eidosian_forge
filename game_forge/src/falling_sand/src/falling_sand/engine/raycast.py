"""Ray casting helpers for voxel interaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class Ray:
    """Ray definition."""

    origin: Vector3
    direction: Vector3

    def __post_init__(self) -> None:
        if self.direction == (0.0, 0.0, 0.0):
            raise ValueError("direction must be non-zero")


@dataclass(frozen=True)
class Plane:
    """Plane definition."""

    point: Vector3
    normal: Vector3

    def __post_init__(self) -> None:
        if self.normal == (0.0, 0.0, 0.0):
            raise ValueError("normal must be non-zero")


def ray_plane_intersect(ray: Ray, plane: Plane) -> Vector3 | None:
    """Return intersection point or None if parallel/behind."""

    ox, oy, oz = ray.origin
    dx, dy, dz = ray.direction
    px, py, pz = plane.point
    nx, ny, nz = plane.normal

    denom = dx * nx + dy * ny + dz * nz
    if denom == 0.0:
        return None

    t = ((px - ox) * nx + (py - oy) * ny + (pz - oz) * nz) / denom
    if t < 0:
        return None

    return (ox + dx * t, oy + dy * t, oz + dz * t)


def point_to_voxel(point: Vector3, voxel_size: float) -> Tuple[int, int, int]:
    """Convert a world-space point to voxel coordinates."""

    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    return (int(point[0] // voxel_size), int(point[1] // voxel_size), int(point[2] // voxel_size))
