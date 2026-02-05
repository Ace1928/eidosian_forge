"""Barnes-Hut tree for approximate N-body forces in 2D or 3D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from algorithms_lab.core import Domain, ensure_f32


@dataclass(frozen=True)
class BarnesHutData:
    """Packed Barnes-Hut tree data."""

    centers: NDArray[np.float32]
    half_sizes: NDArray[np.float32]
    children: NDArray[np.int32]
    masses: NDArray[np.float32]
    com: NDArray[np.float32]
    particle: NDArray[np.int32]


class BarnesHutTree:
    """Barnes-Hut tree builder and evaluator."""

    def __init__(self, domain: Domain, max_depth: int = 32) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        self.domain = domain
        self.max_depth = int(max_depth)
        self._data: BarnesHutData | None = None

    def build(self, positions: NDArray[np.float32], masses: NDArray[np.float32]) -> BarnesHutData:
        """Build Barnes-Hut tree data for positions and masses."""

        pos = ensure_f32(positions)
        mass = np.asarray(masses, dtype=np.float32)
        if pos.shape[0] != mass.shape[0]:
            raise ValueError("positions and masses must have the same length")
        if pos.shape[1] != self.domain.dims:
            raise ValueError("positions do not match domain dimensions")

        center, half_size = self._root_bounds(pos)
        max_children = 4 if self.domain.dims == 2 else 8
        capacity = max(1, pos.shape[0] * (8 if self.domain.dims == 2 else 16))

        centers = np.zeros((capacity, self.domain.dims), dtype=np.float32)
        half_sizes = np.zeros(capacity, dtype=np.float32)
        children = np.full((capacity, max_children), -1, dtype=np.int32)
        node_mass = np.zeros(capacity, dtype=np.float32)
        node_com = np.zeros((capacity, self.domain.dims), dtype=np.float32)
        node_particle = np.full(capacity, -1, dtype=np.int32)

        centers[0] = center
        half_sizes[0] = half_size
        next_node = 1

        for idx in range(pos.shape[0]):
            next_node = self._insert_particle(
                idx,
                pos,
                centers,
                half_sizes,
                children,
                node_particle,
                next_node,
                max_children,
            )
            if next_node >= capacity:
                capacity = int(capacity * 1.5) + 1
                centers = self._grow_array(centers, capacity)
                half_sizes = self._grow_array(half_sizes, capacity)
                children = self._grow_array(children, capacity, fill=-1)
                node_mass = self._grow_array(node_mass, capacity)
                node_com = self._grow_array(node_com, capacity)
                node_particle = self._grow_array(node_particle, capacity, fill=-1)

        centers = centers[:next_node]
        half_sizes = half_sizes[:next_node]
        children = children[:next_node]
        node_mass = node_mass[:next_node]
        node_com = node_com[:next_node]
        node_particle = node_particle[:next_node]

        self._accumulate_mass(pos, mass, children, node_particle, node_mass, node_com)

        data = BarnesHutData(
            centers=centers,
            half_sizes=half_sizes,
            children=children,
            masses=node_mass,
            com=node_com,
            particle=node_particle,
        )
        self._data = data
        return data

    def compute_acceleration(
        self,
        positions: NDArray[np.float32],
        masses: NDArray[np.float32],
        theta: float = 0.5,
        G: float = 1.0,
        softening: float = 1e-3,
    ) -> NDArray[np.float32]:
        """Compute accelerations using Barnes-Hut approximation."""

        if theta <= 0:
            raise ValueError("theta must be positive")
        data = self._data or self.build(positions, masses)
        pos = ensure_f32(positions)
        acc = np.zeros_like(pos)
        for i in range(pos.shape[0]):
            acc[i] = self._accumulate_force(i, pos, data, theta, G, softening)
        return acc

    def _root_bounds(self, positions: NDArray[np.float32]) -> Tuple[NDArray[np.float32], float]:
        """Return a cubic root bound centered on the positions."""

        mins = np.min(positions, axis=0)
        maxs = np.max(positions, axis=0)
        center = 0.5 * (mins + maxs)
        extent = np.max(maxs - mins)
        half_size = max(extent * 0.5, 1e-6)
        return center.astype(np.float32), float(half_size)

    def _insert_particle(
        self,
        idx: int,
        positions: NDArray[np.float32],
        centers: NDArray[np.float32],
        half_sizes: NDArray[np.float32],
        children: NDArray[np.int32],
        node_particle: NDArray[np.int32],
        next_node: int,
        max_children: int,
    ) -> int:
        """Insert a particle index into the tree."""

        node = 0
        depth = 0
        while True:
            particle = int(node_particle[node])
            if particle == -1 and np.all(children[node] == -1):
                node_particle[node] = idx
                return next_node
            if depth >= self.max_depth:
                return next_node
            if particle != -1 and np.all(children[node] == -1):
                node_particle[node] = -1
                next_node = self._subdivide(node, centers, half_sizes, children, next_node)
                child = self._child_index(positions[particle], centers[node])
                child_node = int(children[node, child])
                node_particle[child_node] = particle
            child = self._child_index(positions[idx], centers[node])
            node = int(children[node, child])
            depth += 1

    def _subdivide(
        self,
        node: int,
        centers: NDArray[np.float32],
        half_sizes: NDArray[np.float32],
        children: NDArray[np.int32],
        next_node: int,
    ) -> int:
        """Create children for the node and return updated next index."""

        half = half_sizes[node] * 0.5
        offsets = self._child_offsets()
        for offset in offsets:
            centers[next_node] = centers[node] + offset * half
            half_sizes[next_node] = half
            children[node, self._offset_index(offset)] = next_node
            next_node += 1
        return next_node

    def _child_offsets(self) -> NDArray[np.float32]:
        """Return offsets for children in this dimension."""

        if self.domain.dims == 2:
            return np.array(
                [[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.float32
            )
        return np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [-1, 1, -1],
                [1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [-1, 1, 1],
                [1, 1, 1],
            ],
            dtype=np.float32,
        )

    def _offset_index(self, offset: NDArray[np.float32]) -> int:
        """Map a child offset to its index."""

        if self.domain.dims == 2:
            return 0 if offset[0] < 0 and offset[1] < 0 else 1 if offset[0] > 0 and offset[1] < 0 else 2 if offset[0] < 0 else 3
        idx = 0
        if offset[0] > 0:
            idx |= 1
        if offset[1] > 0:
            idx |= 2
        if offset[2] > 0:
            idx |= 4
        return idx

    def _child_index(self, position: NDArray[np.float32], center: NDArray[np.float32]) -> int:
        """Return child index for a position relative to a node center."""

        if self.domain.dims == 2:
            return 0 if position[0] < center[0] and position[1] < center[1] else 1 if position[0] >= center[0] and position[1] < center[1] else 2 if position[0] < center[0] else 3
        idx = 0
        if position[0] >= center[0]:
            idx |= 1
        if position[1] >= center[1]:
            idx |= 2
        if position[2] >= center[2]:
            idx |= 4
        return idx

    def _accumulate_mass(
        self,
        positions: NDArray[np.float32],
        masses: NDArray[np.float32],
        children: NDArray[np.int32],
        node_particle: NDArray[np.int32],
        node_mass: NDArray[np.float32],
        node_com: NDArray[np.float32],
    ) -> None:
        """Compute masses and centers of mass for all nodes."""

        order = []
        stack = [0]
        while stack:
            node = stack.pop()
            order.append(node)
            for child in children[node]:
                if child != -1:
                    stack.append(int(child))
        for node in reversed(order):
            particle = int(node_particle[node])
            if particle != -1:
                node_mass[node] = masses[particle]
                node_com[node] = positions[particle]
                continue
            child_ids = children[node]
            valid = child_ids[child_ids != -1]
            if valid.size == 0:
                continue
            child_mass = node_mass[valid]
            total_mass = np.sum(child_mass)
            if total_mass > 0:
                node_mass[node] = total_mass
                node_com[node] = np.sum(
                    node_com[valid] * child_mass[:, None], axis=0
                ) / total_mass

    def _accumulate_force(
        self,
        idx: int,
        positions: NDArray[np.float32],
        data: BarnesHutData,
        theta: float,
        G: float,
        softening: float,
    ) -> NDArray[np.float32]:
        """Traverse the tree and accumulate force on one particle."""

        acc = np.zeros(self.domain.dims, dtype=np.float32)
        stack = [0]
        while stack:
            node = stack.pop()
            if data.masses[node] == 0:
                continue
            particle = int(data.particle[node])
            if particle == idx:
                continue
            delta = data.com[node] - positions[idx]
            delta = self.domain.minimal_image(delta)
            dist2 = np.dot(delta, delta) + softening * softening
            dist = float(np.sqrt(dist2))
            size = data.half_sizes[node] * 2.0
            if particle != -1 or size / dist < theta:
                acc += G * data.masses[node] * delta / (dist2 * dist)
            else:
                for child in data.children[node]:
                    if child != -1:
                        stack.append(int(child))
        return acc

    def _grow_array(self, array: NDArray, new_size: int, fill: int | float = 0) -> NDArray:
        """Grow a numpy array along axis 0."""

        new_shape = (new_size,) + array.shape[1:]
        new_array = np.full(new_shape, fill, dtype=array.dtype)
        new_array[: array.shape[0]] = array
        return new_array
