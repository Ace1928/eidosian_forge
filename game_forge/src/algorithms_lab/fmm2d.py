"""Simplified 2D Fast Multipole Method (FMM) using a two-level grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from algorithms_lab.core import Domain, WrapMode, ensure_f32
from algorithms_lab.spatial_hash import UniformGrid


@dataclass(frozen=True)
class FMM2DData:
    """Packed data for a two-level FMM approximation."""

    cell_centers: NDArray[np.float32]
    cell_mass: NDArray[np.float32]
    cell_com: NDArray[np.float32]


class FMM2D:
    """Two-level FMM approximation with near-field direct interactions."""

    def __init__(self, domain: Domain, cell_size: float) -> None:
        if domain.dims != 2:
            raise ValueError("FMM2D requires a 2D domain")
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        self.domain = domain
        self.cell_size = float(cell_size)
        self._grid = UniformGrid(domain, cell_size)
        self._data: FMM2DData | None = None

    def build(self, positions: NDArray[np.float32], masses: NDArray[np.float32]) -> FMM2DData:
        """Build grid aggregates and far-field estimates."""

        pos = ensure_f32(positions)
        mass = np.asarray(masses, dtype=np.float32)
        if pos.shape[0] != mass.shape[0]:
            raise ValueError("positions and masses must have the same length")

        grid = self._grid.build(pos)
        cell_mass = np.bincount(grid.cell_ids, weights=mass, minlength=self._grid.num_cells)
        cell_mass = cell_mass.astype(np.float32)
        sum_pos = np.zeros((self._grid.num_cells, 2), dtype=np.float32)
        weighted_pos = pos * mass[:, None]
        np.add.at(sum_pos, grid.cell_ids, weighted_pos)
        cell_com = np.zeros_like(sum_pos)
        nonzero = cell_mass > 0
        cell_com[nonzero] = sum_pos[nonzero] / cell_mass[nonzero, None]

        cell_centers = self._cell_centers()
        data = FMM2DData(
            cell_centers=cell_centers,
            cell_mass=cell_mass,
            cell_com=cell_com,
        )
        self._data = data
        return data

    def compute_acceleration(
        self,
        positions: NDArray[np.float32],
        masses: NDArray[np.float32],
        G: float = 1.0,
        softening: float = 1e-3,
    ) -> NDArray[np.float32]:
        """Compute accelerations using the two-level FMM approximation."""

        data = self._data or self.build(positions, masses)
        pos = ensure_f32(positions)
        mass = np.asarray(masses, dtype=np.float32)
        grid = self._grid.build(pos)

        acc = np.zeros_like(pos)
        occupied = np.nonzero(data.cell_mass > 0)[0]
        if occupied.size == 0:
            return acc
        occ_mass = data.cell_mass[occupied]
        occ_com = data.cell_com[occupied]
        delta_all = occ_com[None, :, :] - pos[:, None, :]
        delta_all = self.domain.minimal_image(delta_all)
        dist2_all = np.einsum("ijk,ijk->ij", delta_all, delta_all) + softening * softening
        dist_all = np.sqrt(dist2_all)
        acc_all = np.einsum(
            "ij,ijk->ik",
            (occ_mass[None, :] / (dist2_all * dist_all)) * G,
            delta_all,
        )

        # Subtract aggregated near-field cells to avoid double counting.
        lookup = np.full(self._grid.num_cells, -1, dtype=np.int32)
        lookup[occupied] = np.arange(occupied.size, dtype=np.int32)
        coords = self._grid.cell_coords(grid.cell_ids)
        offsets = self._grid.neighbor_offsets()
        neighbor_coords = coords[:, None, :] + offsets[None, :, :]
        if self.domain.wrap == WrapMode.WRAP:
            neighbor_coords = np.mod(neighbor_coords, self._grid.grid_shape)
            valid = np.ones(neighbor_coords.shape[:2], dtype=bool)
        else:
            valid = np.all(
                (neighbor_coords >= 0) & (neighbor_coords < self._grid.grid_shape),
                axis=2,
            )
            neighbor_coords = np.clip(neighbor_coords, 0, self._grid.grid_shape - 1)
        neighbor_ids = self._grid._coords_to_ids(neighbor_coords.reshape(-1, 2)).reshape(
            coords.shape[0], -1
        )
        neighbor_ids = np.where(valid, neighbor_ids, -1)
        neighbor_idx = lookup[neighbor_ids]
        valid = neighbor_idx >= 0
        safe_idx = np.where(valid, neighbor_idx, 0)
        neighbor_mass = occ_mass[safe_idx] * valid
        neighbor_com = occ_com[safe_idx]
        delta_near = neighbor_com - pos[:, None, :]
        delta_near = self.domain.minimal_image(delta_near)
        dist2_near = (
            np.einsum("ijk,ijk->ij", delta_near, delta_near) + softening * softening
        )
        dist_near = np.sqrt(dist2_near)
        acc_near = np.einsum(
            "ij,ijk->ik",
            (neighbor_mass / (dist2_near * dist_near)) * G,
            delta_near,
        )
        acc = acc_all - acc_near

        pairs = self._grid.neighbor_cell_pairs(grid)
        for cell_a, cell_b in pairs:
            idx_a = self._grid.cell_particles(grid, int(cell_a))
            idx_b = self._grid.cell_particles(grid, int(cell_b))
            if idx_a.size == 0 or idx_b.size == 0:
                continue
            if cell_a == cell_b:
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
            dist2 = np.einsum("ij,ij->i", delta, delta) + softening * softening
            dist = np.sqrt(dist2)
            force_i = G * mass[local_j] / (dist2 * dist)
            force_j = G * mass[local_i] / (dist2 * dist)
            np.add.at(acc, local_i, delta * force_i[:, None])
            np.add.at(acc, local_j, -delta * force_j[:, None])

        return acc

    def _cell_centers(self) -> NDArray[np.float32]:
        """Return centers for all grid cells."""

        grid = self._grid.cell_coords(
            np.arange(self._grid.num_cells, dtype=np.int32)
        ).astype(np.float32)
        centers = (grid + 0.5) * self.cell_size + self.domain.mins
        return centers
