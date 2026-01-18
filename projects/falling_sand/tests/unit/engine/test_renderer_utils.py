import pytest

from falling_sand.engine.chunk import Chunk
from falling_sand.engine.materials import Material
from falling_sand.engine import renderer_panda3d
from falling_sand.engine.renderer_panda3d import RenderConfig, build_point_cloud, chunk_points


def test_chunk_points_empty() -> None:
    chunk = Chunk.empty((0, 0, 0), 2)
    positions, materials = chunk_points(chunk, 0.1)
    assert positions.shape == (0, 3)
    assert materials.shape == (0,)


def test_chunk_points_non_empty() -> None:
    chunk = Chunk.empty((1, 0, 0), 2)
    chunk.data[0, 0, 0] = int(Material.GRANULAR)
    positions, materials = chunk_points(chunk, 0.2)
    assert positions.shape == (1, 3)
    assert materials.tolist() == [int(Material.GRANULAR)]
    assert positions[0] == pytest.approx([0.5, 0.1, 0.1])


def test_render_config_validation() -> None:
    with pytest.raises(ValueError, match="point_size must be positive"):
        RenderConfig(point_size=0)
    with pytest.raises(ValueError, match="max_points must be positive"):
        RenderConfig(max_points=0)


def test_build_point_cloud_missing_dependency() -> None:
    chunk = Chunk.empty((0, 0, 0), 1)
    if renderer_panda3d.GeomVertexFormat is not None:
        pytest.skip("Panda3D available; point cloud should build.")
    with pytest.raises(RuntimeError, match="Panda3D is not available"):
        build_point_cloud([chunk], config=None, palette=None, render_config=None)  # type: ignore[arg-type]
