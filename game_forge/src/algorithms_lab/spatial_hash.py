"""Uniform grid / spatial hashing utilities for neighbor queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from algorithms_lab.core import Domain, WrapMode, ensure_i32


@dataclass(frozen=True)
class GridData:
    """Packed grid data for neighbor queries."""

    cell_size: float
    grid_shape: NDArray[np.int32]
    cell_ids: NDArray[np.int32]
    sorted_indices: NDArray[np.int32]
    cell_start: NDArray[np.int32]
    cell_count: NDArray[np.int32]


class UniformGrid:
    """Uniform grid for fast neighbor searches in 2D or 3D."""

    def __init__(self, domain: Domain, cell_size: float) -> None:
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        self.domain = domain
        self.cell_size = float(cell_size)
        grid_shape = np.ceil(domain.sizes / cell_size).astype(np.int32)
        self.grid_shape = np.maximum(grid_shape, 1).astype(np.int32)
        self.num_cells = int(np.prod(self.grid_shape))
        self._neighbor_offsets = self._build_neighbor_offsets()

    def _build_neighbor_offsets(self) -> NDArray[np.int32]:
        """Return offsets for neighbor cells including the origin cell."""

        ranges = [np.array([-1, 0, 1], dtype=np.int32) for _ in range(self.domain.dims)]
        mesh = np.meshgrid(*ranges, indexing="ij")
        offsets = np.stack([axis.reshape(-1) for axis in mesh], axis=-1)
        return offsets.astype(np.int32)

    def build(self, positions: NDArray[np.float32]) -> GridData:
        """Build grid data from particle positions."""

        pos = np.asarray(positions, dtype=np.float32)
        if pos.shape[-1] != self.domain.dims:
            raise ValueError("Positions do not match domain dimensions")
        coords = np.floor((pos - self.domain.mins) / self.cell_size).astype(np.int32)
        if self.domain.wrap == WrapMode.WRAP:
            coords = np.mod(coords, self.grid_shape)
        else:
            coords = np.minimum(np.maximum(coords, 0), self.grid_shape - 1)
        cell_ids = self._coords_to_ids(coords)
        order = np.argsort(cell_ids, kind="stable")
        sorted_ids = cell_ids[order]
        cell_count = np.bincount(sorted_ids, minlength=self.num_cells).astype(np.int32)
        cell_start = np.cumsum(cell_count, dtype=np.int32) - cell_count
        return GridData(
            cell_size=self.cell_size,
            grid_shape=self.grid_shape,
            cell_ids=ensure_i32(cell_ids),
            sorted_indices=ensure_i32(order),
            cell_start=ensure_i32(cell_start),
            cell_count=ensure_i32(cell_count),
        )

    def neighbor_offsets(self) -> NDArray[np.int32]:
        """Return cached neighbor offsets for the grid."""

        return self._neighbor_offsets.copy()

    def occupied_cells(self, grid: GridData) -> NDArray[np.int32]:
        """Return ids of occupied cells."""

        return np.nonzero(grid.cell_count)[0].astype(np.int32)

    def cell_coords(self, cell_ids: NDArray[np.int32]) -> NDArray[np.int32]:
        """Convert cell ids to coordinates."""

        return np.stack(np.unravel_index(cell_ids, self.grid_shape), axis=-1).astype(
            np.int32
        )

    def neighbor_cell_pairs(self, grid: GridData) -> NDArray[np.int32]:
        """Return pairs of occupied neighbor cells (upper triangle only)."""

        occupied = self.occupied_cells(grid)
        if occupied.size == 0:
            return np.zeros((0, 2), dtype=np.int32)
        coords = self.cell_coords(occupied)
        pairs = []
        for offset in self._neighbor_offsets:
            neighbor_coords = coords + offset
            if self.domain.wrap == WrapMode.WRAP:
                neighbor_coords = np.mod(neighbor_coords, self.grid_shape)
            else:
                valid = np.all(
                    (neighbor_coords >= 0) & (neighbor_coords < self.grid_shape), axis=1
                )
                neighbor_coords = neighbor_coords[valid]
                occupied_subset = occupied[valid]
            if self.domain.wrap == WrapMode.WRAP:
                occupied_subset = occupied
            neighbor_ids = self._coords_to_ids(neighbor_coords)
            if neighbor_ids.size == 0:
                continue
            # Keep only non-decreasing pairs to avoid duplicates.
            mask = neighbor_ids >= occupied_subset
            if np.any(mask):
                pairs.append(
                    np.stack([occupied_subset[mask], neighbor_ids[mask]], axis=-1)
                )
        if not pairs:
            return np.zeros((0, 2), dtype=np.int32)
        return np.concatenate(pairs, axis=0).astype(np.int32)

    def _coords_to_ids(self, coords: NDArray[np.int32]) -> NDArray[np.int32]:
        """Convert integer cell coordinates to ids."""

        if coords.shape[-1] != self.domain.dims:
            raise ValueError("Cell coordinates have wrong dimensionality")
        if self.domain.dims == 2:
            return coords[:, 0] + self.grid_shape[0] * coords[:, 1]
        return coords[:, 0] + self.grid_shape[0] * (
            coords[:, 1] + self.grid_shape[1] * coords[:, 2]
        )

    def cell_particles(
        self, grid: GridData, cell_id: int
    ) -> NDArray[np.int32]:
        """Return sorted particle indices for a given cell id."""

        count = int(grid.cell_count[cell_id])
        if count == 0:
            return np.zeros(0, dtype=np.int32)
        start = int(grid.cell_start[cell_id])
        return grid.sorted_indices[start : start + count]

    def neighbor_pairs(
        self,
        positions: NDArray[np.float32],
        radius: float,
    ) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Return neighbor pairs within radius using the uniform grid."""

        if radius <= 0:
            raise ValueError("radius must be positive")
        grid = self.build(positions)
        pairs = self.neighbor_cell_pairs(grid)
        if pairs.size == 0:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
        pos = np.asarray(positions, dtype=np.float32)
        radius2 = radius * radius
        pair_i = []
        pair_j = []
        for cell_a, cell_b in pairs:
            idx_a = self.cell_particles(grid, int(cell_a))
            idx_b = self.cell_particles(grid, int(cell_b))
            if idx_a.size == 0 or idx_b.size == 0:
                continue
            if cell_a == cell_b:
                # Upper triangle within the same cell.
                if idx_a.size < 2:
                    continue
                ia, ib = np.triu_indices(idx_a.size, k=1)
                local_i = idx_a[ia]
                local_j = idx_a[ib]
            else:
                local_i = np.repeat(idx_a, idx_b.size)
                local_j = np.tile(idx_b, idx_a.size)
            delta = pos[local_j] - pos[local_i]
            delta = self.domain.minimal_image(delta)
            dist2 = np.einsum("ij,ij->i", delta, delta)
            mask = dist2 <= radius2
            if np.any(mask):
                pair_i.append(local_i[mask])
                pair_j.append(local_j[mask])
        if not pair_i:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
        return ensure_i32(np.concatenate(pair_i)), ensure_i32(np.concatenate(pair_j))
