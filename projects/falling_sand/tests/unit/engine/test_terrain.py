import numpy as np
import pytest

from falling_sand.engine.materials import Material
from falling_sand.engine.terrain import (
    TerrainConfig,
    TerrainGenerator,
    fractal_noise_2d,
    generate_chunk,
    hash2d,
    heightmap_for_chunk,
    value_noise_2d,
)


def test_terrain_config_validation() -> None:
    with pytest.raises(ValueError, match="size_x must be positive"):
        TerrainConfig(size_x=0)
    with pytest.raises(ValueError, match="size_y must be positive"):
        TerrainConfig(size_y=0)
    with pytest.raises(ValueError, match="height_layers must be positive"):
        TerrainConfig(height_layers=0)
    with pytest.raises(ValueError, match="water_level must be non-negative"):
        TerrainConfig(water_level=-1)
    with pytest.raises(ValueError, match="soil_depth must be positive"):
        TerrainConfig(soil_depth=0)
    with pytest.raises(ValueError, match="base_height must be non-negative"):
        TerrainConfig(base_height=-1)
    with pytest.raises(ValueError, match="height_variation must be positive"):
        TerrainConfig(height_variation=0.0)
    with pytest.raises(ValueError, match="noise_scale must be positive"):
        TerrainConfig(noise_scale=0.0)
    with pytest.raises(ValueError, match="octaves must be positive"):
        TerrainConfig(octaves=0)
    with pytest.raises(ValueError, match="persistence must be positive"):
        TerrainConfig(persistence=0.0)
    with pytest.raises(ValueError, match="lacunarity must be positive"):
        TerrainConfig(lacunarity=0.0)
    with pytest.raises(ValueError, match="seed must be non-negative"):
        TerrainConfig(seed=-1)
    with pytest.raises(ValueError, match="grid_cache must be positive"):
        TerrainConfig(grid_cache=0)


def test_hash2d_repeatable() -> None:
    x = np.array([[0, 1], [2, 3]])
    y = np.array([[0, 1], [2, 3]])
    a = hash2d(x, y, seed=42)
    b = hash2d(x, y, seed=42)
    assert np.allclose(a, b)
    assert np.all((a >= 0.0) & (a < 1.0))


def test_value_noise_2d_shape() -> None:
    x = np.array([[0.0, 0.5], [1.0, 1.5]])
    y = np.array([[0.0, 0.5], [1.0, 1.5]])
    noise = value_noise_2d(x, y, seed=1)
    assert noise.shape == x.shape


def test_fractal_noise_2d_bounds() -> None:
    config = TerrainConfig()
    x = np.array([[0.0, 0.5], [1.0, 1.5]])
    y = np.array([[0.0, 0.5], [1.0, 1.5]])
    noise = fractal_noise_2d(x, y, config)
    assert np.all((noise >= 0.0) & (noise <= 1.0))


def test_heightmap_for_chunk() -> None:
    config = TerrainConfig(height_layers=10, base_height=2, height_variation=1.0)
    heightmap = heightmap_for_chunk((0, 0, 0), 4, config)
    assert heightmap.shape == (4, 4)
    assert heightmap.min() >= 0
    assert heightmap.max() <= config.height_layers - 1


def test_generate_chunk_bounds() -> None:
    config = TerrainConfig(size_x=8, size_y=8, height_layers=4, water_level=1)
    empty = generate_chunk((2, 0, 0), 4, config)
    assert empty.is_empty()


def test_generate_chunk_materials() -> None:
    config = TerrainConfig(
        size_x=4,
        size_y=4,
        height_layers=4,
        base_height=1,
        height_variation=0.1,
        water_level=2,
        soil_depth=1,
    )
    chunk = generate_chunk((0, 0, 0), 4, config)
    assert chunk.data.dtype == np.uint8
    assert np.any(chunk.data == int(Material.SOLID))
    assert np.any(chunk.data == int(Material.GRANULAR))
    assert np.any(chunk.data == int(Material.LIQUID))


def test_terrain_generator_provider() -> None:
    generator = TerrainGenerator(TerrainConfig(size_x=8, size_y=8, height_layers=4))
    chunk = generator.chunk((0, 0, 0), 4)
    assert chunk.size == 4
