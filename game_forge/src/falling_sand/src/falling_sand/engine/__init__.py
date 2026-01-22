"""Falling sand engine core modules."""

from falling_sand.engine.config import VoxelConfig
from falling_sand.engine.materials import Material, MaterialPalette
from falling_sand.engine.world import World
from falling_sand.engine.raycast import Plane, Ray, point_to_voxel, ray_plane_intersect
from falling_sand.engine.simulation import SimulationConfig, StepStats, step_world
from falling_sand.engine.streaming import ChunkStreamer, StreamConfig
from falling_sand.engine.terrain import TerrainConfig, TerrainGenerator, generate_chunk, heightmap_for_chunk
from falling_sand.engine.tools import erase_sphere, place_sphere
from falling_sand.engine.ui import OverlayConfig, UiOverlay
from falling_sand.engine.renderer_instancing import InstanceConfig, InstancedVoxelRenderer, iter_instances

__all__ = [
    "Material",
    "MaterialPalette",
    "VoxelConfig",
    "World",
    "SimulationConfig",
    "StepStats",
    "step_world",
    "Ray",
    "Plane",
    "ray_plane_intersect",
    "point_to_voxel",
    "ChunkStreamer",
    "StreamConfig",
    "TerrainConfig",
    "TerrainGenerator",
    "generate_chunk",
    "heightmap_for_chunk",
    "InstancedVoxelRenderer",
    "InstanceConfig",
    "iter_instances",
    "place_sphere",
    "erase_sphere",
    "UiOverlay",
    "OverlayConfig",
]
