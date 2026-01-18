import numpy as np
import pytest

from falling_sand.engine.chunk import Chunk
from falling_sand.engine.materials import Material


def test_chunk_empty() -> None:
    chunk = Chunk.empty((0, 0, 0), 4)
    assert chunk.size == 4
    assert chunk.is_empty()


def test_chunk_get_set() -> None:
    chunk = Chunk.empty((0, 0, 0), 2)
    chunk.set(0, 0, 0, Material.SOLID)
    assert chunk.get(0, 0, 0) == Material.SOLID
    assert not chunk.is_empty()


def test_chunk_validation() -> None:
    with pytest.raises(ValueError, match="Chunk data must be 3D"):
        Chunk(coord=(0, 0, 0), data=np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="Chunk data must be uint8"):
        Chunk(coord=(0, 0, 0), data=np.zeros((2, 2, 2), dtype=np.int32))


def test_chunk_copy() -> None:
    chunk = Chunk.empty((0, 0, 0), 2)
    chunk.set(1, 1, 1, Material.LIQUID)
    copy = chunk.copy()
    assert copy.get(1, 1, 1) == Material.LIQUID
    assert copy is not chunk
