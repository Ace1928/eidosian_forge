"""Numba-accelerated Barnes-Hut traversal."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from algorithms_lab.backends import HAS_NUMBA, njit, prange

if HAS_NUMBA:

    @njit(cache=True)
    def _minimal_image(dx: float, size: float, inv_size: float) -> float:
        return dx - size * np.round(dx * inv_size)


    @njit(cache=True, fastmath=True)
    def _accumulate_force_single(
        idx: int,
        positions: NDArray[np.float32],
        centers: NDArray[np.float32],
        half_sizes: NDArray[np.float32],
        children: NDArray[np.int32],
        node_mass: NDArray[np.float32],
        node_com: NDArray[np.float32],
        node_particle: NDArray[np.int32],
        theta: float,
        G: float,
        softening: float,
        sizes: NDArray[np.float32],
        inv_sizes: NDArray[np.float32],
        wrap: int,
        dims: int,
    ) -> NDArray[np.float32]:
        num_nodes = centers.shape[0]
        stack = np.empty(num_nodes, dtype=np.int32)
        top = 0
        stack[top] = 0
        top += 1
        ax = 0.0
        ay = 0.0
        az = 0.0
        while top > 0:
            top -= 1
            node = stack[top]
            mass = node_mass[node]
            if mass == 0.0:
                continue
            particle = node_particle[node]
            if particle == idx and particle != -1:
                continue
            dx = node_com[node, 0] - positions[idx, 0]
            dy = node_com[node, 1] - positions[idx, 1]
            dz = 0.0
            if wrap == 1:
                dx = _minimal_image(dx, sizes[0], inv_sizes[0])
                dy = _minimal_image(dy, sizes[1], inv_sizes[1])
            if dims == 3:
                dz = node_com[node, 2] - positions[idx, 2]
                if wrap == 1:
                    dz = _minimal_image(dz, sizes[2], inv_sizes[2])
            dist2 = dx * dx + dy * dy + (dz * dz if dims == 3 else 0.0) + softening * softening
            dist = np.sqrt(dist2)
            size = half_sizes[node] * 2.0
            if particle != -1 or size / dist < theta:
                inv = G * mass / (dist2 * dist)
                ax += dx * inv
                ay += dy * inv
                if dims == 3:
                    az += dz * inv
            else:
                for c in range(children.shape[1]):
                    child = children[node, c]
                    if child != -1:
                        stack[top] = child
                        top += 1
        out = np.zeros(dims, dtype=np.float32)
        out[0] = ax
        out[1] = ay
        if dims == 3:
            out[2] = az
        return out


    @njit(cache=True, parallel=True, fastmath=True)
    def barnes_hut_accel(
        positions: NDArray[np.float32],
        centers: NDArray[np.float32],
        half_sizes: NDArray[np.float32],
        children: NDArray[np.int32],
        node_mass: NDArray[np.float32],
        node_com: NDArray[np.float32],
        node_particle: NDArray[np.int32],
        theta: float,
        G: float,
        softening: float,
        sizes: NDArray[np.float32],
        inv_sizes: NDArray[np.float32],
        wrap: int,
        dims: int,
    ) -> NDArray[np.float32]:
        n = positions.shape[0]
        acc = np.zeros_like(positions)
        for i in prange(n):
            acc[i] = _accumulate_force_single(
                i,
                positions,
                centers,
                half_sizes,
                children,
                node_mass,
                node_com,
                node_particle,
                theta,
                G,
                softening,
                sizes,
                inv_sizes,
                wrap,
                dims,
            )
        return acc

else:

    def barnes_hut_accel(*args, **kwargs):  # type: ignore
        raise ImportError("numba is not installed")
