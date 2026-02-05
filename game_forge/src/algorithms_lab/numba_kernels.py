"""Numba-accelerated kernels for neighbor pair enumeration."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from algorithms_lab.backends import HAS_NUMBA, njit

if HAS_NUMBA:

    @njit(cache=True)
    def _count_pairs(
        cell_pairs: NDArray[np.int32],
        cell_start: NDArray[np.int32],
        cell_count: NDArray[np.int32],
        sorted_indices: NDArray[np.int32],
        positions: NDArray[np.float32],
        radius2: float,
        sizes: NDArray[np.float32],
        inv_sizes: NDArray[np.float32],
        wrap: int,
        dims: int,
    ) -> int:
        total = 0
        for pair_idx in range(cell_pairs.shape[0]):
            cell_a = cell_pairs[pair_idx, 0]
            cell_b = cell_pairs[pair_idx, 1]
            count_a = cell_count[cell_a]
            count_b = cell_count[cell_b]
            if count_a == 0 or count_b == 0:
                continue
            start_a = cell_start[cell_a]
            start_b = cell_start[cell_b]
            if cell_a == cell_b:
                for i_local in range(count_a - 1):
                    i = sorted_indices[start_a + i_local]
                    for j_local in range(i_local + 1, count_a):
                        j = sorted_indices[start_a + j_local]
                        dx = positions[j, 0] - positions[i, 0]
                        dy = positions[j, 1] - positions[i, 1]
                        dz = 0.0
                        if wrap == 1:
                            dx -= sizes[0] * np.round(dx * inv_sizes[0])
                            dy -= sizes[1] * np.round(dy * inv_sizes[1])
                            if dims == 3:
                                dz = positions[j, 2] - positions[i, 2]
                                dz -= sizes[2] * np.round(dz * inv_sizes[2])
                        elif dims == 3:
                            dz = positions[j, 2] - positions[i, 2]
                        dist2 = dx * dx + dy * dy + (dz * dz if dims == 3 else 0.0)
                        if dist2 <= radius2:
                            total += 1
            else:
                for i_local in range(count_a):
                    i = sorted_indices[start_a + i_local]
                    for j_local in range(count_b):
                        j = sorted_indices[start_b + j_local]
                        dx = positions[j, 0] - positions[i, 0]
                        dy = positions[j, 1] - positions[i, 1]
                        dz = 0.0
                        if wrap == 1:
                            dx -= sizes[0] * np.round(dx * inv_sizes[0])
                            dy -= sizes[1] * np.round(dy * inv_sizes[1])
                            if dims == 3:
                                dz = positions[j, 2] - positions[i, 2]
                                dz -= sizes[2] * np.round(dz * inv_sizes[2])
                        elif dims == 3:
                            dz = positions[j, 2] - positions[i, 2]
                        dist2 = dx * dx + dy * dy + (dz * dz if dims == 3 else 0.0)
                        if dist2 <= radius2:
                            total += 1
        return total

    @njit(cache=True)
    def _fill_pairs(
        cell_pairs: NDArray[np.int32],
        cell_start: NDArray[np.int32],
        cell_count: NDArray[np.int32],
        sorted_indices: NDArray[np.int32],
        positions: NDArray[np.float32],
        radius2: float,
        sizes: NDArray[np.float32],
        inv_sizes: NDArray[np.float32],
        wrap: int,
        dims: int,
        pair_i: NDArray[np.int32],
        pair_j: NDArray[np.int32],
    ) -> int:
        idx = 0
        for pair_idx in range(cell_pairs.shape[0]):
            cell_a = cell_pairs[pair_idx, 0]
            cell_b = cell_pairs[pair_idx, 1]
            count_a = cell_count[cell_a]
            count_b = cell_count[cell_b]
            if count_a == 0 or count_b == 0:
                continue
            start_a = cell_start[cell_a]
            start_b = cell_start[cell_b]
            if cell_a == cell_b:
                for i_local in range(count_a - 1):
                    i = sorted_indices[start_a + i_local]
                    for j_local in range(i_local + 1, count_a):
                        j = sorted_indices[start_a + j_local]
                        dx = positions[j, 0] - positions[i, 0]
                        dy = positions[j, 1] - positions[i, 1]
                        dz = 0.0
                        if wrap == 1:
                            dx -= sizes[0] * np.round(dx * inv_sizes[0])
                            dy -= sizes[1] * np.round(dy * inv_sizes[1])
                            if dims == 3:
                                dz = positions[j, 2] - positions[i, 2]
                                dz -= sizes[2] * np.round(dz * inv_sizes[2])
                        elif dims == 3:
                            dz = positions[j, 2] - positions[i, 2]
                        dist2 = dx * dx + dy * dy + (dz * dz if dims == 3 else 0.0)
                        if dist2 <= radius2:
                            pair_i[idx] = i
                            pair_j[idx] = j
                            idx += 1
            else:
                for i_local in range(count_a):
                    i = sorted_indices[start_a + i_local]
                    for j_local in range(count_b):
                        j = sorted_indices[start_b + j_local]
                        dx = positions[j, 0] - positions[i, 0]
                        dy = positions[j, 1] - positions[i, 1]
                        dz = 0.0
                        if wrap == 1:
                            dx -= sizes[0] * np.round(dx * inv_sizes[0])
                            dy -= sizes[1] * np.round(dy * inv_sizes[1])
                            if dims == 3:
                                dz = positions[j, 2] - positions[i, 2]
                                dz -= sizes[2] * np.round(dz * inv_sizes[2])
                        elif dims == 3:
                            dz = positions[j, 2] - positions[i, 2]
                        dist2 = dx * dx + dy * dy + (dz * dz if dims == 3 else 0.0)
                        if dist2 <= radius2:
                            pair_i[idx] = i
                            pair_j[idx] = j
                            idx += 1
        return idx


    def neighbor_pairs_numba(
        cell_pairs: NDArray[np.int32],
        cell_start: NDArray[np.int32],
        cell_count: NDArray[np.int32],
        sorted_indices: NDArray[np.int32],
        positions: NDArray[np.float32],
        radius: float,
        sizes: NDArray[np.float32],
        inv_sizes: NDArray[np.float32],
        wrap: bool,
        dims: int,
    ) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        if cell_pairs.size == 0:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
        radius2 = float(radius * radius)
        total = _count_pairs(
            cell_pairs,
            cell_start,
            cell_count,
            sorted_indices,
            positions,
            radius2,
            sizes,
            inv_sizes,
            1 if wrap else 0,
            dims,
        )
        if total == 0:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
        pair_i = np.empty(total, dtype=np.int32)
        pair_j = np.empty(total, dtype=np.int32)
        filled = _fill_pairs(
            cell_pairs,
            cell_start,
            cell_count,
            sorted_indices,
            positions,
            radius2,
            sizes,
            inv_sizes,
            1 if wrap else 0,
            dims,
            pair_i,
            pair_j,
        )
        return pair_i[:filled], pair_j[:filled]

else:

    def neighbor_pairs_numba(*args, **kwargs):  # type: ignore
        raise ImportError("numba is not installed")
