import pytest

from falling_sand.engine.coords import chunk_origin, local_to_world, world_to_chunk


def test_world_to_chunk_round_trip() -> None:
    voxel = (12, 5, 3)
    chunk, local = world_to_chunk(voxel, 10)
    assert chunk == (1, 0, 0)
    assert local == (2, 5, 3)
    assert local_to_world(chunk, local, 10) == voxel


def test_chunk_origin() -> None:
    assert chunk_origin((2, -1, 0), 10) == (20, -10, 0)


def test_world_to_chunk_invalid_size() -> None:
    with pytest.raises(ZeroDivisionError):
        world_to_chunk((1, 2, 3), 0)
