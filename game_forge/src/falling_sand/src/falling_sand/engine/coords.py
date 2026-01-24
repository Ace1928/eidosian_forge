"""Coordinate utilities for chunked voxel grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from eidosian_core import eidosian


VoxelCoord = Tuple[int, int, int]
ChunkCoord = Tuple[int, int, int]
LocalCoord = Tuple[int, int, int]


@dataclass(frozen=True)
class CoordConfig:
    """Coordinate configuration for chunking."""

    chunk_size: int

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")


@eidosian()
def world_to_chunk(voxel: VoxelCoord, chunk_size: int) -> tuple[ChunkCoord, LocalCoord]:
    """Convert world voxel coordinates to chunk and local coordinates."""

    cx = voxel[0] // chunk_size
    cy = voxel[1] // chunk_size
    cz = voxel[2] // chunk_size
    lx = voxel[0] - cx * chunk_size
    ly = voxel[1] - cy * chunk_size
    lz = voxel[2] - cz * chunk_size
    return (cx, cy, cz), (lx, ly, lz)


@eidosian()
def chunk_origin(chunk: ChunkCoord, chunk_size: int) -> VoxelCoord:
    """Return the world voxel origin for a chunk."""

    return (chunk[0] * chunk_size, chunk[1] * chunk_size, chunk[2] * chunk_size)


@eidosian()
def local_to_world(chunk: ChunkCoord, local: LocalCoord, chunk_size: int) -> VoxelCoord:
    """Convert local chunk coordinates to world voxel coordinates."""

    origin = chunk_origin(chunk, chunk_size)
    return (origin[0] + local[0], origin[1] + local[1], origin[2] + local[2])
