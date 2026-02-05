"""Multi-level fast multipole-style solver using hierarchical cell lists."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from algorithms_lab.core import Domain, WrapMode, ensure_f32
from algorithms_lab.spatial_hash import UniformGrid


@dataclass(frozen=True)
class MultiLevelFMMData:
    """Packed data for the multi-level FMM solver."""

    level_mass: List[NDArray[np.float32]]
    level_com: List[NDArray[np.float32]]
    level_center: List[NDArray[np.float32]]
    level_local: List[NDArray[np.float32]]
    leaf_ids: NDArray[np.int32]
    leaf_cell_size: NDArray[np.float32]


class MultiLevelFMM:
    """Multi-level FMM-style solver with monopole far field and direct near field."""

    def __init__(
        self,
        domain: Domain,
        levels: int = 4,
        G: float = 1.0,
        softening: float = 1e-3,
    ) -> None:
        if levels < 1:
            raise ValueError("levels must be at least 1")
        self.domain = domain
        self.levels = int(levels)
        self.G = float(G)
        self.softening = float(softening)
        self._data: MultiLevelFMMData | None = None
        self._interaction_offsets = self._build_interaction_offsets()

    def build(self, positions: NDArray[np.float32], masses: NDArray[np.float32]) -> MultiLevelFMMData:
        """Build the multi-level cell hierarchy."""

        pos = ensure_f32(positions)
        mass = np.asarray(masses, dtype=np.float32)
        if pos.shape[0] != mass.shape[0]:
            raise ValueError("positions and masses must have the same length")
        if pos.shape[1] != self.domain.dims:
            raise ValueError("positions do not match domain dimensions")

        dims = self.domain.dims
        level_mass: List[NDArray[np.float32]] = []
        level_com: List[NDArray[np.float32]] = []
        level_center: List[NDArray[np.float32]] = []
        level_local: List[NDArray[np.float32]] = []

        grid_shapes = [np.array([2**l] * dims, dtype=np.int32) for l in range(self.levels + 1)]
        cell_sizes = [self.domain.sizes / shape for shape in grid_shapes]

        leaf_shape = grid_shapes[-1]
        leaf_cell_size = cell_sizes[-1].astype(np.float32)
        leaf_coords = np.floor((pos - self.domain.mins) / leaf_cell_size).astype(np.int32)
        if self.domain.wrap == WrapMode.WRAP:
            leaf_coords = np.mod(leaf_coords, leaf_shape)
        else:
            leaf_coords = np.minimum(np.maximum(leaf_coords, 0), leaf_shape - 1)
        leaf_ids = self._coords_to_ids(leaf_coords, leaf_shape)

        num_leaf = int(np.prod(leaf_shape))
        leaf_mass = np.bincount(leaf_ids, weights=mass, minlength=num_leaf).astype(np.float32)
        sum_pos = np.zeros((num_leaf, dims), dtype=np.float32)
        weighted = pos * mass[:, None]
        for axis in range(dims):
            sum_pos[:, axis] = np.bincount(
                leaf_ids, weights=weighted[:, axis], minlength=num_leaf
            ).astype(np.float32)
        leaf_com = np.zeros_like(sum_pos)
        nonzero = leaf_mass > 0
        leaf_com[nonzero] = sum_pos[nonzero] / leaf_mass[nonzero, None]

        level_mass.append(leaf_mass)
        level_com.append(leaf_com)
        level_center.append(self._cell_centers(leaf_shape, leaf_cell_size))
        level_local.append(np.zeros_like(leaf_com))

        # Upward pass: aggregate from leaf to root.
        current_mass = leaf_mass
        current_sum = sum_pos
        for level in range(self.levels - 1, -1, -1):
            parent_shape = grid_shapes[level]
            parent_mass, parent_sum = self._aggregate_to_parent(
                current_mass, current_sum, parent_shape
            )
            parent_com = np.zeros_like(parent_sum)
            nonzero = parent_mass > 0
            parent_com[nonzero] = parent_sum[nonzero] / parent_mass[nonzero, None]
            level_mass.append(parent_mass)
            level_com.append(parent_com)
            level_center.append(self._cell_centers(parent_shape, cell_sizes[level]))
            level_local.append(np.zeros_like(parent_com))
            current_mass = parent_mass
            current_sum = parent_sum

        # Reverse so level 0 is root.
        level_mass = list(reversed(level_mass))
        level_com = list(reversed(level_com))
        level_center = list(reversed(level_center))
        level_local = list(reversed(level_local))

        data = MultiLevelFMMData(
            level_mass=level_mass,
            level_com=level_com,
            level_center=level_center,
            level_local=level_local,
            leaf_ids=leaf_ids.astype(np.int32),
            leaf_cell_size=leaf_cell_size,
        )
        self._data = data
        return data

    def compute_acceleration(
        self, positions: NDArray[np.float32], masses: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Compute accelerations using the multi-level FMM solver."""

        data = self._data or self.build(positions, masses)
        pos = ensure_f32(positions)
        mass = np.asarray(masses, dtype=np.float32)
        dims = self.domain.dims

        # Reset local fields.
        for level_local in data.level_local:
            level_local.fill(0.0)

        # Far-field contributions per level.
        for level in range(1, self.levels + 1):
            self._accumulate_far_field(level, data)

        # Downward pass: propagate parent local fields to children.
        for level in range(1, self.levels + 1):
            data.level_local[level] += self._expand_to_children(
                data.level_local[level - 1], level
            )

        # Start with far-field at leaf.
        acc = data.level_local[-1][data.leaf_ids].copy()

        # Direct near-field interactions at leaf level.
        grid = UniformGrid(self.domain, cell_size=data.leaf_cell_size[0])
        grid_data = grid.build(pos)
        pairs = grid.neighbor_cell_pairs(grid_data)
        for cell_a, cell_b in pairs:
            idx_a = grid.cell_particles(grid_data, int(cell_a))
            idx_b = grid.cell_particles(grid_data, int(cell_b))
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
            dist2 = np.einsum("ij,ij->i", delta, delta) + self.softening * self.softening
            dist = np.sqrt(dist2)
            force_i = self.G * mass[local_j] / (dist2 * dist)
            force_j = self.G * mass[local_i] / (dist2 * dist)
            np.add.at(acc, local_i, delta * force_i[:, None])
            np.add.at(acc, local_j, -delta * force_j[:, None])

        return acc

    def _aggregate_to_parent(
        self,
        mass_child: NDArray[np.float32],
        sum_child: NDArray[np.float32],
        parent_shape: NDArray[np.int32],
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Aggregate child masses and sums into parent cells."""

        dims = self.domain.dims
        if dims == 2:
            ny, nx = int(parent_shape[1] * 2), int(parent_shape[0] * 2)
            mass_grid = mass_child.reshape(ny, nx)
            sum_grid = sum_child.reshape(ny, nx, dims)
            mass_parent = mass_grid.reshape(ny // 2, 2, nx // 2, 2).sum(axis=(1, 3))
            sum_parent = sum_grid.reshape(ny // 2, 2, nx // 2, 2, dims).sum(axis=(1, 3))
            return mass_parent.reshape(-1), sum_parent.reshape(-1, dims)
        nz, ny, nx = (
            int(parent_shape[2] * 2),
            int(parent_shape[1] * 2),
            int(parent_shape[0] * 2),
        )
        mass_grid = mass_child.reshape(nz, ny, nx)
        sum_grid = sum_child.reshape(nz, ny, nx, dims)
        mass_parent = mass_grid.reshape(nz // 2, 2, ny // 2, 2, nx // 2, 2).sum(
            axis=(1, 3, 5)
        )
        sum_parent = sum_grid.reshape(
            nz // 2, 2, ny // 2, 2, nx // 2, 2, dims
        ).sum(axis=(1, 3, 5))
        return mass_parent.reshape(-1), sum_parent.reshape(-1, dims)

    def _cell_centers(self, grid_shape: NDArray[np.int32], cell_size: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return centers for all cells in a level."""

        coords = self._cell_coords(np.arange(int(np.prod(grid_shape)), dtype=np.int32), grid_shape)
        centers = (coords.astype(np.float32) + 0.5) * cell_size + self.domain.mins
        return centers

    def _accumulate_far_field(self, level: int, data: MultiLevelFMMData) -> None:
        """Accumulate far-field contributions for a given level."""

        dims = self.domain.dims
        grid_shape = np.array([2**level] * dims, dtype=np.int32)
        coords = self._cell_coords(
            np.arange(int(np.prod(grid_shape)), dtype=np.int32), grid_shape
        )
        centers = data.level_center[level]
        level_mass = data.level_mass[level]
        level_com = data.level_com[level]
        local = data.level_local[level]

        parities = coords % 2
        parity_map = self._interaction_offsets
        for parity_idx, offsets in parity_map.items():
            if offsets.size == 0:
                continue
            parity = np.array(parity_idx, dtype=np.int32)
            mask = np.all(parities == parity, axis=1)
            if not np.any(mask):
                continue
            target_ids = np.nonzero(mask)[0]
            target_coords = coords[mask]
            target_centers = centers[target_ids]
            for offset in offsets:
                src_coords = target_coords + offset
                if self.domain.wrap == WrapMode.WRAP:
                    src_coords = np.mod(src_coords, grid_shape)
                    valid = np.ones(src_coords.shape[0], dtype=bool)
                else:
                    valid = np.all((src_coords >= 0) & (src_coords < grid_shape), axis=1)
                    src_coords = np.clip(src_coords, 0, grid_shape - 1)
                if not np.any(valid):
                    continue
                src_ids = self._coords_to_ids(src_coords[valid], grid_shape)
                masses = level_mass[src_ids]
                nonzero = masses > 0
                if not np.any(nonzero):
                    continue
                src_ids = src_ids[nonzero]
                tgt_ids = target_ids[valid][nonzero]
                tgt_centers = target_centers[valid][nonzero]
                com = level_com[src_ids]
                delta = com - tgt_centers
                delta = self.domain.minimal_image(delta)
                dist2 = np.einsum("ij,ij->i", delta, delta) + self.softening * self.softening
                dist = np.sqrt(dist2)
                inv = self.G * level_mass[src_ids] / (dist2 * dist)
                contrib = delta * inv[:, None]
                for axis in range(dims):
                    np.add.at(local[:, axis], tgt_ids, contrib[:, axis])

    def _expand_to_children(self, parent_local: NDArray[np.float32], level: int) -> NDArray[np.float32]:
        """Expand parent local field to child cells for a given level."""

        dims = self.domain.dims
        grid_shape = np.array([2**level] * dims, dtype=np.int32)
        if dims == 2:
            ny, nx = grid_shape[1], grid_shape[0]
            parent = parent_local.reshape(ny // 2, nx // 2, dims)
            expanded = parent.repeat(2, axis=0).repeat(2, axis=1)
            return expanded.reshape(-1, dims)
        nz, ny, nx = grid_shape[2], grid_shape[1], grid_shape[0]
        parent = parent_local.reshape(nz // 2, ny // 2, nx // 2, dims)
        expanded = parent.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
        return expanded.reshape(-1, dims)

    def _coords_to_ids(self, coords: NDArray[np.int32], grid_shape: NDArray[np.int32]) -> NDArray[np.int32]:
        """Convert coordinates to ids for a given grid shape."""

        if coords.shape[-1] != self.domain.dims:
            raise ValueError("Coordinate dimension mismatch")
        if self.domain.dims == 2:
            return coords[:, 0] + grid_shape[0] * coords[:, 1]
        return coords[:, 0] + grid_shape[0] * (coords[:, 1] + grid_shape[1] * coords[:, 2])

    def _cell_coords(self, cell_ids: NDArray[np.int32], grid_shape: NDArray[np.int32]) -> NDArray[np.int32]:
        """Convert ids to coordinates for a given grid shape."""

        cell_ids = np.asarray(cell_ids, dtype=np.int32)
        if self.domain.dims == 2:
            x = cell_ids % grid_shape[0]
            y = cell_ids // grid_shape[0]
            return np.stack([x, y], axis=-1).astype(np.int32)
        stride = grid_shape[0] * grid_shape[1]
        x = cell_ids % grid_shape[0]
        y = (cell_ids // grid_shape[0]) % grid_shape[1]
        z = cell_ids // stride
        return np.stack([x, y, z], axis=-1).astype(np.int32)

    def _build_interaction_offsets(self) -> Dict[Tuple[int, ...], NDArray[np.int32]]:
        """Build interaction list offsets for each parity pattern."""

        dims = self.domain.dims
        parent_offsets = list(product([-1, 0, 1], repeat=dims))
        child_offsets = list(product([0, 1], repeat=dims))
        parities = list(product([0, 1], repeat=dims))
        offsets_by_parity: Dict[Tuple[int, ...], NDArray[np.int32]] = {}
        for parity in parities:
            offsets = set()
            for po in parent_offsets:
                for co in child_offsets:
                    offset = tuple(2 * np.array(po) + np.array(co) - np.array(parity))
                    if all(-1 <= o <= 1 for o in offset):
                        continue
                    offsets.add(offset)
            if offsets:
                offsets_by_parity[parity] = np.array(sorted(offsets), dtype=np.int32)
            else:
                offsets_by_parity[parity] = np.zeros((0, dims), dtype=np.int32)
        return offsets_by_parity
