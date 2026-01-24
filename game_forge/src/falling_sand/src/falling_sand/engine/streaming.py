"""Chunk streaming utilities for large worlds."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable, Set

from falling_sand.engine.chunk import Chunk
from falling_sand.engine.coords import ChunkCoord, VoxelCoord, world_to_chunk
from falling_sand.engine.world import World
from eidosian_core import eidosian


ChunkProvider = Callable[[ChunkCoord, int], Chunk]


@dataclass(frozen=True)
class StreamConfig:
    """Configuration for chunk streaming."""

    radius: int = 1
    cache_limit: int = 0

    def __post_init__(self) -> None:
        if self.radius < 0:
            raise ValueError("radius must be non-negative")
        if self.cache_limit < 0:
            raise ValueError("cache_limit must be non-negative")


@eidosian()
def default_provider(coord: ChunkCoord, size: int) -> Chunk:
    """Create a default empty chunk."""

    return Chunk.empty(coord, size)


@eidosian()
def coords_in_radius(center: ChunkCoord, radius: int) -> Iterable[ChunkCoord]:
    """Yield chunk coords within a cubic radius."""

    if radius < 0:
        raise ValueError("radius must be non-negative")
    cx, cy, cz = center
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                yield (cx + dx, cy + dy, cz + dz)


class ChunkStreamer:
    """Maintain active chunks around a focus point."""

    def __init__(
        self,
        world: World,
        config: StreamConfig | None = None,
        provider: ChunkProvider | None = None,
    ) -> None:
        self.world = world
        self.config = config or StreamConfig()
        self.provider = provider or default_provider
        self.active: Set[ChunkCoord] = set()
        self._cache: "OrderedDict[ChunkCoord, Chunk]" = OrderedDict()

    @eidosian()
    def focus_from_voxel(self, voxel: VoxelCoord) -> ChunkCoord:
        """Return chunk coordinate for a voxel coordinate."""

        chunk, _ = world_to_chunk(voxel, self.world.config.chunk_size_voxels)
        return chunk

    @eidosian()
    def update_focus(self, focus: ChunkCoord) -> None:
        """Update active chunks around a focus chunk coordinate."""

        desired = set(coords_in_radius(focus, self.config.radius))
        to_add = desired - self.active
        to_remove = self.active - desired

        for coord in to_add:
            if self.world.get_chunk(coord) is None:
                chunk = self._cache.pop(coord, None)
                if chunk is None:
                    chunk = self.provider(coord, self.world.config.chunk_size_voxels)
                self.world.chunks[coord] = chunk
        for coord in to_remove:
            chunk = self.world.get_chunk(coord)
            if chunk is not None:
                del self.world.chunks[coord]
                if self.config.cache_limit > 0:
                    self._cache[coord] = chunk
                    if len(self._cache) > self.config.cache_limit:
                        self._cache.popitem(last=False)

        self.active = desired

    @eidosian()
    def update_focus_voxel(self, voxel: VoxelCoord) -> None:
        """Update active chunks based on a voxel coordinate."""

        self.update_focus(self.focus_from_voxel(voxel))
