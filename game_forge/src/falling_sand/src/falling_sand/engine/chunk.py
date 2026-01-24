"""Chunk storage for voxel materials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from falling_sand.engine.materials import Material
from eidosian_core import eidosian


ChunkCoord = Tuple[int, int, int]


@dataclass
class Chunk:
    """A single chunk of voxel data."""

    coord: ChunkCoord
    data: np.ndarray
    dirty: bool = False

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError("Chunk data must be 3D")
        if self.data.dtype != np.uint8:
            raise ValueError("Chunk data must be uint8")

    @property
    def size(self) -> int:
        """Return chunk size along one axis."""

        return int(self.data.shape[0])

    @classmethod
    def empty(cls, coord: ChunkCoord, size: int) -> "Chunk":
        """Create an empty chunk with AIR voxels."""

        if size <= 0:
            raise ValueError("size must be positive")
        data = np.full((size, size, size), Material.AIR, dtype=np.uint8)
        return cls(coord=coord, data=data)

    @eidosian()
    def get(self, x: int, y: int, z: int) -> Material:
        """Get material at local voxel coordinates."""

        return Material(int(self.data[x, y, z]))

    @eidosian()
    def set(self, x: int, y: int, z: int, material: Material) -> None:
        """Set material at local voxel coordinates."""

        self.data[x, y, z] = int(material)
        self.dirty = True

    @eidosian()
    def is_empty(self) -> bool:
        """Return True if all voxels are AIR."""

        return bool(np.all(self.data == int(Material.AIR)))

    @eidosian()
    def copy(self) -> "Chunk":
        """Return a copy of the chunk."""

        return Chunk(coord=self.coord, data=self.data.copy(), dirty=self.dirty)
