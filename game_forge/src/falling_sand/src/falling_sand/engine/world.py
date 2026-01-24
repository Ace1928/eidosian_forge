"""World container for chunked voxel data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

import numpy as np

from falling_sand.engine.chunk import Chunk
from falling_sand.engine.config import VoxelConfig
from falling_sand.engine.coords import ChunkCoord, VoxelCoord, world_to_chunk
from falling_sand.engine.materials import Material
from eidosian_core import eidosian


@dataclass
class World:
    """Chunked voxel world."""

    config: VoxelConfig
    chunks: Dict[ChunkCoord, Chunk] = field(default_factory=dict)

    @eidosian()
    def get_chunk(self, coord: ChunkCoord) -> Chunk | None:
        """Return chunk at coord if present."""

        return self.chunks.get(coord)

    @eidosian()
    def ensure_chunk(self, coord: ChunkCoord) -> Chunk:
        """Get or create a chunk at the given coordinate."""

        chunk = self.chunks.get(coord)
        if chunk is None:
            chunk = Chunk.empty(coord, self.config.chunk_size_voxels)
            self.chunks[coord] = chunk
        return chunk

    @eidosian()
    def iter_chunks(self) -> Iterable[Chunk]:
        """Iterate over chunks in the world."""

        return self.chunks.values()

    @eidosian()
    def get_voxel(self, voxel: VoxelCoord) -> Material:
        """Get material at world voxel coordinates."""

        chunk_coord, local = world_to_chunk(voxel, self.config.chunk_size_voxels)
        chunk = self.chunks.get(chunk_coord)
        if chunk is None:
            return Material.AIR
        return chunk.get(*local)

    @eidosian()
    def set_voxel(self, voxel: VoxelCoord, material: Material) -> None:
        """Set material at world voxel coordinates."""

        chunk_coord, local = world_to_chunk(voxel, self.config.chunk_size_voxels)
        chunk = self.ensure_chunk(chunk_coord)
        chunk.set(*local, material)

    @eidosian()
    def fill_chunk(self, coord: ChunkCoord, material: Material) -> None:
        """Fill an entire chunk with a material."""

        chunk = self.ensure_chunk(coord)
        chunk.data.fill(int(material))
        chunk.dirty = True

    @eidosian()
    def prune_empty(self) -> None:
        """Remove chunks that are empty to save memory."""

        to_remove = [coord for coord, chunk in self.chunks.items() if chunk.is_empty()]
        for coord in to_remove:
            del self.chunks[coord]

    @eidosian()
    def snapshot(self) -> Dict[ChunkCoord, np.ndarray]:
        """Return a copy of chunk data arrays."""

        return {coord: chunk.data.copy() for coord, chunk in self.chunks.items()}

    @eidosian()
    def surface_height(self, x: int, y: int, default: int = 0) -> int:
        """Return the highest solid voxel at (x, y) or default."""

        max_height = default
        for coord, chunk in self.chunks.items():
            origin_x = coord[0] * self.config.chunk_size_voxels
            origin_y = coord[1] * self.config.chunk_size_voxels
            local_x = x - origin_x
            local_y = y - origin_y
            if not (0 <= local_x < chunk.size and 0 <= local_y < chunk.size):
                continue
            column = chunk.data[local_x, local_y, :]
            solid_indices = np.where(column == int(Material.SOLID))[0]
            if solid_indices.size == 0:
                continue
            top = int(solid_indices.max()) + coord[2] * self.config.chunk_size_voxels
            if top > max_height:
                max_height = top
        return max_height
