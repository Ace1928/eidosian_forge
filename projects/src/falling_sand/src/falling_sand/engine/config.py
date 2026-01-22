"""Configuration for the falling sand engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoxelConfig:
    """Simulation grid configuration."""

    voxel_size_m: float = 0.1
    chunk_size_voxels: int = 10

    def __post_init__(self) -> None:
        if self.voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        if self.chunk_size_voxels <= 0:
            raise ValueError("chunk_size_voxels must be positive")

    @property
    def chunk_size_m(self) -> float:
        """Physical size of a chunk in meters."""

        return self.voxel_size_m * float(self.chunk_size_voxels)
