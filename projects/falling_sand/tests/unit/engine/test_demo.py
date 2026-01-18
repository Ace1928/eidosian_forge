import numpy as np
import pytest

from falling_sand.engine.config import VoxelConfig
from falling_sand.engine.demo import (
    DemoApp,
    DemoConfig,
    clamp_position,
    ground_provider,
    orbit_offset,
    player_height,
    run_demo,
    spawn_materials,
)
from falling_sand.engine.materials import Material
from falling_sand.engine import renderer_instancing
from falling_sand.engine.terrain import TerrainConfig
from falling_sand.engine.world import World


def test_demo_config_validation() -> None:
    with pytest.raises(ValueError, match="spawn_height must be non-negative"):
        DemoConfig(spawn_height=-1)
    with pytest.raises(ValueError, match="spawn_radius must be positive"):
        DemoConfig(spawn_radius=0)
    with pytest.raises(ValueError, match="spawn_count must be positive"):
        DemoConfig(spawn_count=0)
    with pytest.raises(ValueError, match="step_interval must be positive"):
        DemoConfig(step_interval=0)
    with pytest.raises(ValueError, match="render_interval must be positive"):
        DemoConfig(render_interval=0)
    with pytest.raises(ValueError, match="camera_distance_m must be positive"):
        DemoConfig(camera_distance_m=0)
    with pytest.raises(ValueError, match="camera_min_distance_m must be positive"):
        DemoConfig(camera_min_distance_m=0)
    with pytest.raises(ValueError, match="camera_max_distance_m must be greater than camera_min_distance_m"):
        DemoConfig(camera_min_distance_m=5.0, camera_max_distance_m=4.0)
    with pytest.raises(ValueError, match="camera_min_pitch_deg must be less than camera_max_pitch_deg"):
        DemoConfig(camera_min_pitch_deg=80.0, camera_max_pitch_deg=10.0)
    with pytest.raises(ValueError, match="mouse_sensitivity must be positive"):
        DemoConfig(mouse_sensitivity=0.0)
    with pytest.raises(ValueError, match="zoom_step_m must be positive"):
        DemoConfig(zoom_step_m=0.0)
    with pytest.raises(ValueError, match="yaw_step_deg must be positive"):
        DemoConfig(yaw_step_deg=0.0)
    with pytest.raises(ValueError, match="stream_radius must be non-negative"):
        DemoConfig(stream_radius=-1)
    with pytest.raises(ValueError, match="spawn_position must be XY"):
        DemoConfig(spawn_position=(1,))
    with pytest.raises(ValueError, match="spawn_position x must be within terrain bounds"):
        DemoConfig(spawn_position=(-1, 0))
    with pytest.raises(ValueError, match="spawn_position y must be within terrain bounds"):
        DemoConfig(spawn_position=(0, 10_000))
    with pytest.raises(ValueError, match="spawn_height must be below terrain height_layers"):
        DemoConfig(spawn_height=100, terrain_config=TerrainConfig(height_layers=10))
    with pytest.raises(ValueError, match="instance_max must be positive"):
        DemoConfig(instance_max=0)
    with pytest.raises(ValueError, match="stream_cache must be non-negative"):
        DemoConfig(stream_cache=-1)


def test_spawn_materials() -> None:
    world = World(config=VoxelConfig())
    config = DemoConfig(spawn_count=5)
    rng = np.random.default_rng(0)
    count = spawn_materials(world, rng, Material.GRANULAR, config)
    assert count == 5
    count = spawn_materials(world, rng, Material.GRANULAR, config, center=(10, 10))
    assert count == 5


def test_run_demo_missing_dependency() -> None:
    if renderer_instancing.ShowBase is not None:
        pytest.skip("Panda3D available; demo should run.")
    with pytest.raises(RuntimeError, match="Panda3D is not available"):
        run_demo()


def test_ground_provider() -> None:
    chunk = ground_provider((0, 0, 0), 4)
    assert chunk.data[:, :, 0].min() == int(Material.SOLID)
    assert chunk.data[:, :, 0].max() == int(Material.SOLID)
    other = ground_provider((0, 0, 1), 4)
    assert other.is_empty()


def test_orbit_offset() -> None:
    offset = orbit_offset(0.0, 0.0, 1.0)
    assert offset == (0.0, 1.0, 0.0)
    offset = orbit_offset(90.0, 0.0, 2.0)
    assert pytest.approx(offset[0], rel=1e-6) == 2.0
    assert pytest.approx(offset[1], rel=1e-6) == 0.0


def test_player_height() -> None:
    world = World(config=VoxelConfig())
    world.set_voxel((0, 0, 0), Material.SOLID)
    assert player_height(world, 0, 0, 1) == 1
    assert player_height(world, 0, 0, 5) == 5


def test_demo_clamp() -> None:
    assert DemoApp._clamp(5.0, 1.0, 4.0) == 4.0
    assert DemoApp._clamp(-1.0, 0.0, 4.0) == 0.0


def test_clamp_position() -> None:
    terrain = TerrainConfig(size_x=10, size_y=10)
    assert clamp_position(-1, 5, terrain) == (0, 5)
    assert clamp_position(9, 20, terrain) == (9, 9)
