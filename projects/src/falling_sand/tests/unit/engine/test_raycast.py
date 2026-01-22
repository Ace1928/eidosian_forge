import pytest

from falling_sand.engine.raycast import Plane, Ray, point_to_voxel, ray_plane_intersect


def test_ray_plane_intersect() -> None:
    ray = Ray(origin=(0.0, 0.0, 1.0), direction=(0.0, 0.0, -1.0))
    plane = Plane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
    hit = ray_plane_intersect(ray, plane)
    assert hit == (0.0, 0.0, 0.0)


def test_ray_plane_parallel() -> None:
    ray = Ray(origin=(0.0, 0.0, 1.0), direction=(1.0, 0.0, 0.0))
    plane = Plane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
    assert ray_plane_intersect(ray, plane) is None


def test_point_to_voxel() -> None:
    assert point_to_voxel((0.25, 0.35, 0.0), 0.1) == (2, 3, 0)


def test_point_to_voxel_invalid() -> None:
    with pytest.raises(ValueError, match="voxel_size must be positive"):
        point_to_voxel((0.0, 0.0, 0.0), 0.0)
