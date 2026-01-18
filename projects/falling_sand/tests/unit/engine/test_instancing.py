import pytest
from falling_sand.engine.materials import Material
from falling_sand.engine.renderer_instancing import (
    InstanceConfig,
    chunk_bounding_radius,
    chunk_center,
    frustum_visible,
    iter_chunk_instances,
    iter_instances,
    within_distance,
)
from falling_sand.engine.chunk import Chunk


def test_instance_config_validation() -> None:
    with pytest.raises(ValueError, match="max_instances must be positive"):
        InstanceConfig(max_instances=0)
    with pytest.raises(ValueError, match="cube_model must be non-empty"):
        InstanceConfig(cube_model="")
    with pytest.raises(ValueError, match="max_view_distance_m must be non-negative"):
        InstanceConfig(max_view_distance_m=-1.0)


def test_iter_instances() -> None:
    chunk = Chunk.empty((0, 0, 0), 2)
    chunk.data[0, 0, 0] = int(Material.GRANULAR)
    items = list(iter_instances([chunk], 0.1))
    assert items[0][:3] == (0.05, 0.05, 0.05)
    assert items[0][3] == int(Material.GRANULAR)


def test_iter_chunk_instances_offset() -> None:
    chunk = Chunk.empty((1, 0, 0), 2)
    chunk.data[0, 0, 0] = int(Material.GRANULAR)
    items = list(iter_chunk_instances(chunk, 1.0))
    assert items[0][:3] == (2.5, 0.5, 0.5)


def test_chunk_center_radius() -> None:
    center = chunk_center((1, 0, 0), 2, 1.0)
    assert center == (3.0, 1.0, 1.0)
    radius = chunk_bounding_radius(2, 1.0)
    assert radius > 0.0


def test_within_distance() -> None:
    assert within_distance((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0)
    assert not within_distance((2.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0)


def test_frustum_visible() -> None:
    visible = frustum_visible(
        center=(0.0, 5.0, 0.0),
        radius=0.5,
        camera_pos=(0.0, 0.0, 0.0),
        camera_forward=(0.0, 1.0, 0.0),
        camera_up=(0.0, 0.0, 1.0),
        fov_y_deg=90.0,
        aspect=1.0,
        near=0.1,
        far=10.0,
    )
    assert visible
    hidden = frustum_visible(
        center=(0.0, -5.0, 0.0),
        radius=0.5,
        camera_pos=(0.0, 0.0, 0.0),
        camera_forward=(0.0, 1.0, 0.0),
        camera_up=(0.0, 0.0, 1.0),
        fov_y_deg=90.0,
        aspect=1.0,
        near=0.1,
        far=10.0,
    )
    assert not hidden
