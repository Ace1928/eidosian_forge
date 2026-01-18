import pytest

from falling_sand.engine.config import VoxelConfig
from falling_sand.engine.streaming import ChunkStreamer, StreamConfig, coords_in_radius
from falling_sand.engine.world import World


def test_coords_in_radius() -> None:
    coords = list(coords_in_radius((0, 0, 0), 1))
    assert (1, 1, 1) in coords
    assert (-1, -1, -1) in coords


def test_coords_in_radius_invalid() -> None:
    with pytest.raises(ValueError, match="radius must be non-negative"):
        list(coords_in_radius((0, 0, 0), -1))
    with pytest.raises(ValueError, match="cache_limit must be non-negative"):
        StreamConfig(cache_limit=-1)


def test_chunk_streamer_updates_world() -> None:
    world = World(config=VoxelConfig())
    streamer = ChunkStreamer(world, StreamConfig(radius=1))
    streamer.update_focus((0, 0, 0))
    assert len(world.chunks) == 27
    streamer.update_focus((3, 0, 0))
    assert len(world.chunks) == 27


def test_chunk_streamer_cache() -> None:
    world = World(config=VoxelConfig())
    streamer = ChunkStreamer(world, StreamConfig(radius=0, cache_limit=2))
    streamer.update_focus((0, 0, 0))
    chunk = world.get_chunk((0, 0, 0))
    assert chunk is not None
    streamer.update_focus((1, 0, 0))
    assert (0, 0, 0) in streamer._cache
    streamer.update_focus((0, 0, 0))
    assert world.get_chunk((0, 0, 0)) is chunk


def test_chunk_streamer_focus_voxel() -> None:
    world = World(config=VoxelConfig())
    streamer = ChunkStreamer(world, StreamConfig(radius=0))
    streamer.update_focus_voxel((5, 5, 5))
    assert (0, 0, 0) in world.chunks
