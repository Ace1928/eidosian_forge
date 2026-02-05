"""Core domain types and vector utilities for Algorithms Lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class WrapMode(str, Enum):
    """Boundary behavior for spatial domains."""

    WRAP = "wrap"
    CLAMP = "clamp"
    NONE = "none"


@dataclass(frozen=True)
class Domain:
    """Axis-aligned domain supporting 2D or 3D periodic or clamped boundaries."""

    mins: NDArray[np.float32]
    maxs: NDArray[np.float32]
    wrap: WrapMode = WrapMode.WRAP
    _sizes: NDArray[np.float32] = field(init=False, repr=False)
    _inv_sizes: NDArray[np.float32] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mins = np.asarray(self.mins, dtype=np.float32)
        maxs = np.asarray(self.maxs, dtype=np.float32)
        if mins.shape != maxs.shape:
            raise ValueError("Domain mins/maxs must share the same shape")
        if mins.ndim != 1 or mins.size not in (2, 3):
            raise ValueError("Domain must be 2D or 3D")
        if np.any(maxs <= mins):
            raise ValueError("Domain maxs must be greater than mins")
        sizes = (maxs - mins).astype(np.float32)
        inv_sizes = (1.0 / sizes).astype(np.float32)
        object.__setattr__(self, "mins", mins)
        object.__setattr__(self, "maxs", maxs)
        object.__setattr__(self, "_sizes", sizes)
        object.__setattr__(self, "_inv_sizes", inv_sizes)

    @property
    def dims(self) -> int:
        """Return the number of spatial dimensions."""

        return int(self.mins.size)

    @property
    def sizes(self) -> NDArray[np.float32]:
        """Return the domain extent for each axis."""

        return self._sizes

    @property
    def inv_sizes(self) -> NDArray[np.float32]:
        """Return cached inverse domain sizes."""

        return self._inv_sizes

    def apply_boundary(self, positions: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply boundary conditions to positions and return a new array."""

        pos = ensure_f32(positions)
        if pos.shape[-1] != self.dims:
            raise ValueError("Positions have incompatible dimension")
        if self.wrap == WrapMode.WRAP:
            return self.wrap_positions(pos)
        if self.wrap == WrapMode.CLAMP:
            return self.clamp_positions(pos)
        return pos.copy()

    def wrap_positions(self, positions: NDArray[np.float32]) -> NDArray[np.float32]:
        """Wrap positions into the domain using periodic boundaries."""

        pos = ensure_f32(positions)
        return ((pos - self.mins) % self._sizes) + self.mins

    def clamp_positions(self, positions: NDArray[np.float32]) -> NDArray[np.float32]:
        """Clamp positions into the domain bounds."""

        pos = ensure_f32(positions)
        return np.minimum(np.maximum(pos, self.mins), self.maxs)

    def minimal_image(self, delta: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply minimal-image convention for wrapped domains."""

        if self.wrap != WrapMode.WRAP:
            return delta
        return delta - self._sizes * np.round(delta * self._inv_sizes)


def ensure_f32(array: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return a contiguous float32 array."""

    return np.ascontiguousarray(array, dtype=np.float32)


def ensure_i32(array: NDArray[np.integer]) -> NDArray[np.int32]:
    """Return a contiguous int32 array."""

    return np.ascontiguousarray(array, dtype=np.int32)


def axis_aligned_bounds(
    positions: NDArray[np.float32], padding: float = 0.0
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute axis-aligned bounds for positions with optional padding."""

    pos = np.asarray(positions, dtype=np.float32)
    mins = np.min(pos, axis=0) - padding
    maxs = np.max(pos, axis=0) + padding
    return mins.astype(np.float32), maxs.astype(np.float32)
