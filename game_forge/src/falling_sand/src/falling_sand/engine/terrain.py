"""Procedural terrain generation for large voxel worlds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from falling_sand.engine.chunk import Chunk
from falling_sand.engine.materials import Material
from eidosian_core import eidosian


@dataclass(frozen=True)
class TerrainConfig:
    """Configuration for procedural terrain generation."""

    size_x: int = 10_000
    size_y: int = 10_000
    height_layers: int = 20
    water_level: int = 6
    soil_depth: int = 2
    base_height: int = 6
    height_variation: float = 8.0
    noise_scale: float = 0.005
    octaves: int = 4
    persistence: float = 0.5
    lacunarity: float = 2.0
    seed: int = 1337
    grid_cache: int = 4

    def __post_init__(self) -> None:
        if self.size_x <= 0:
            raise ValueError("size_x must be positive")
        if self.size_y <= 0:
            raise ValueError("size_y must be positive")
        if self.height_layers <= 0:
            raise ValueError("height_layers must be positive")
        if self.water_level < 0:
            raise ValueError("water_level must be non-negative")
        if self.soil_depth <= 0:
            raise ValueError("soil_depth must be positive")
        if self.base_height < 0:
            raise ValueError("base_height must be non-negative")
        if self.height_variation <= 0:
            raise ValueError("height_variation must be positive")
        if self.noise_scale <= 0:
            raise ValueError("noise_scale must be positive")
        if self.octaves <= 0:
            raise ValueError("octaves must be positive")
        if self.persistence <= 0:
            raise ValueError("persistence must be positive")
        if self.lacunarity <= 0:
            raise ValueError("lacunarity must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.grid_cache <= 0:
            raise ValueError("grid_cache must be positive")


@eidosian()
def hash2d(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Hash integer grid coordinates into a repeatable [0, 1) float noise."""

    x_u = np.asarray(x, dtype=np.uint64)
    y_u = np.asarray(y, dtype=np.uint64)
    seed_u = np.uint64(seed)
    h = x_u * np.uint64(0x9E3779B185EBCA87) + y_u * np.uint64(0xC2B2AE3D27D4EB4F) + seed_u
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xFF51AFD7ED558CCD)
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xC4CEB9FE1A85EC53)
    h ^= h >> np.uint64(33)
    return (h & np.uint64((1 << 53) - 1)).astype(np.float64) / float(1 << 53)


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


@eidosian()
def value_noise_2d(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Value noise in 2D using bilinear interpolation."""

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    xf = x - x0
    yf = y - y0

    v00 = hash2d(x0, y0, seed)
    v10 = hash2d(x1, y0, seed)
    v01 = hash2d(x0, y1, seed)
    v11 = hash2d(x1, y1, seed)

    u = _fade(xf)
    v = _fade(yf)

    x_interp0 = v00 * (1.0 - u) + v10 * u
    x_interp1 = v01 * (1.0 - u) + v11 * u
    return x_interp0 * (1.0 - v) + x_interp1 * v


@eidosian()
def fractal_noise_2d(x: np.ndarray, y: np.ndarray, config: TerrainConfig) -> np.ndarray:
    """Compute fractal value noise over multiple octaves."""

    amplitude = 1.0
    frequency = 1.0
    x_b, y_b = np.broadcast_arrays(x, y)
    total = np.zeros_like(x_b, dtype=np.float64)
    max_amplitude = 0.0
    for _ in range(config.octaves):
        total += value_noise_2d(x_b * frequency, y_b * frequency, config.seed) * amplitude
        max_amplitude += amplitude
        amplitude *= config.persistence
        frequency *= config.lacunarity
    if max_amplitude == 0.0:
        return total
    return total / max_amplitude


@eidosian()
def heightmap_for_chunk(coord: tuple[int, int, int], size: int, config: TerrainConfig) -> np.ndarray:
    """Generate a heightmap for a chunk coordinate."""

    if size <= 0:
        raise ValueError("size must be positive")

    x0 = coord[0] * size
    y0 = coord[1] * size
    xs = np.arange(size, dtype=np.int64) + x0
    ys = np.arange(size, dtype=np.int64) + y0
    xg = xs[:, None]
    yg = ys[None, :]

    nx = xg.astype(np.float64) * config.noise_scale
    ny = yg.astype(np.float64) * config.noise_scale
    noise = fractal_noise_2d(nx, ny, config)

    height = config.base_height + noise * config.height_variation
    height = np.clip(height, 0, config.height_layers - 1)
    return height.astype(np.int32)


@eidosian()
def generate_chunk(coord: tuple[int, int, int], size: int, config: TerrainConfig) -> Chunk:
    """Generate a chunk using procedural terrain rules."""

    if size <= 0:
        raise ValueError("size must be positive")

    x0 = coord[0] * size
    y0 = coord[1] * size
    z0 = coord[2] * size
    x1 = x0 + size
    y1 = y0 + size
    z1 = z0 + size

    if x1 <= 0 or y1 <= 0 or x0 >= config.size_x or y0 >= config.size_y:
        return Chunk.empty(coord, size)

    max_layer = max(config.height_layers - 1, config.water_level)
    if z0 > max_layer or z1 <= 0:
        return Chunk.empty(coord, size)

    heightmap = heightmap_for_chunk(coord, size, config)
    data = np.zeros((size, size, size), dtype=np.uint8)

    z_indices = np.arange(size, dtype=np.int64) + z0
    z = z_indices[None, None, :]
    heights = heightmap[:, :, None]
    ground = z <= heights

    soil_start = heights - (config.soil_depth - 1)
    soil_start = np.maximum(soil_start, 0)
    granular_mask = ground & (z >= soil_start)
    solid_mask = ground & ~granular_mask
    water_mask = (z > heights) & (z <= config.water_level)

    data[solid_mask] = int(Material.SOLID)
    data[granular_mask] = int(Material.GRANULAR)
    data[water_mask] = int(Material.LIQUID)

    chunk = Chunk.empty(coord, size)
    if np.any(data != int(Material.AIR)):
        chunk.data = data
        chunk.dirty = True
    return chunk


@dataclass
class TerrainGenerator:
    """Chunk provider wrapper for procedural terrain."""

    config: TerrainConfig
    _grid_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict,
        init=False,
    )

    @eidosian()
    def chunk(self, coord: tuple[int, int, int], size: int) -> Chunk:
        """Return a generated chunk for the given coordinate."""

        return self._generate_chunk_cached(coord, size)

    def _generate_chunk_cached(self, coord: tuple[int, int, int], size: int) -> Chunk:
        if size <= 0:
            raise ValueError("size must be positive")

        x0 = coord[0] * size
        y0 = coord[1] * size
        z0 = coord[2] * size
        x1 = x0 + size
        y1 = y0 + size
        z1 = z0 + size

        if x1 <= 0 or y1 <= 0 or x0 >= self.config.size_x or y0 >= self.config.size_y:
            return Chunk.empty(coord, size)

        max_layer = max(self.config.height_layers - 1, self.config.water_level)
        if z0 > max_layer or z1 <= 0:
            return Chunk.empty(coord, size)

        heightmap = self._heightmap(coord, size)
        data = np.zeros((size, size, size), dtype=np.uint8)

        z_indices = self._grid(size)[2] + z0
        z = z_indices[None, None, :]
        heights = heightmap[:, :, None]
        ground = z <= heights

        soil_start = heights - (self.config.soil_depth - 1)
        soil_start = np.maximum(soil_start, 0)
        granular_mask = ground & (z >= soil_start)
        solid_mask = ground & ~granular_mask
        water_mask = (z > heights) & (z <= self.config.water_level)

        data[solid_mask] = int(Material.SOLID)
        data[granular_mask] = int(Material.GRANULAR)
        data[water_mask] = int(Material.LIQUID)

        chunk = Chunk.empty(coord, size)
        if np.any(data != int(Material.AIR)):
            chunk.data = data
            chunk.dirty = True
        return chunk

    def _grid(self, size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._grid_cache.get(size)
        if cached is not None:
            return cached
        xs = np.arange(size, dtype=np.int64)
        ys = np.arange(size, dtype=np.int64)
        zs = np.arange(size, dtype=np.int64)
        self._grid_cache[size] = (xs, ys, zs)
        if len(self._grid_cache) > self.config.grid_cache:
            self._grid_cache.pop(next(iter(self._grid_cache)))
        return self._grid_cache[size]

    def _heightmap(self, coord: tuple[int, int, int], size: int) -> np.ndarray:
        x0 = coord[0] * size
        y0 = coord[1] * size
        xs, ys, _ = self._grid(size)
        xg = (xs + x0)[:, None]
        yg = (ys + y0)[None, :]

        nx = xg.astype(np.float64) * self.config.noise_scale
        ny = yg.astype(np.float64) * self.config.noise_scale
        noise = fractal_noise_2d(nx, ny, self.config)

        height = self.config.base_height + noise * self.config.height_variation
        height = np.clip(height, 0, self.config.height_layers - 1)
        return height.astype(np.int32)
