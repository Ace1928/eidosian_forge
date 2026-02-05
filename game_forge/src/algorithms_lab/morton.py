"""Morton (Z-order) encoding and sorting for spatial locality."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from algorithms_lab.core import Domain, ensure_i32


def _part1by1(n: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Spread 16-bit integers so there is one zero bit between each bit."""

    n = n & np.uint32(0x0000FFFF)
    n = (n | (n << np.uint32(8))) & np.uint32(0x00FF00FF)
    n = (n | (n << np.uint32(4))) & np.uint32(0x0F0F0F0F)
    n = (n | (n << np.uint32(2))) & np.uint32(0x33333333)
    n = (n | (n << np.uint32(1))) & np.uint32(0x55555555)
    return n


def _part1by2(n: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Spread 10-bit integers so there are two zero bits between each bit."""

    n = n & np.uint32(0x000003FF)
    n = (n | (n << np.uint32(16))) & np.uint32(0x030000FF)
    n = (n | (n << np.uint32(8))) & np.uint32(0x0300F00F)
    n = (n | (n << np.uint32(4))) & np.uint32(0x030C30C3)
    n = (n | (n << np.uint32(2))) & np.uint32(0x09249249)
    return n


def morton_encode(positions: NDArray[np.float32], domain: Domain, bits: int = 10) -> NDArray[np.uint32]:
    """Encode positions into Morton codes for 2D or 3D domains."""

    if bits < 1 or bits > 16:
        raise ValueError("bits must be in [1, 16]")
    if domain.dims not in (2, 3):
        raise ValueError("Domain must be 2D or 3D")
    pos = np.asarray(positions, dtype=np.float32)
    if pos.shape[-1] != domain.dims:
        raise ValueError("Positions do not match domain dimensions")
    scale = (2**bits - 1) / domain.sizes
    normalized = (pos - domain.mins) * scale
    grid = np.clip(np.floor(normalized), 0, 2**bits - 1).astype(np.uint32)
    if domain.dims == 2:
        x = _part1by1(grid[:, 0])
        y = _part1by1(grid[:, 1])
        return x | (y << np.uint32(1))
    x = _part1by2(grid[:, 0])
    y = _part1by2(grid[:, 1])
    z = _part1by2(grid[:, 2])
    return x | (y << np.uint32(1)) | (z << np.uint32(2))


def morton_sort(
    positions: NDArray[np.float32], domain: Domain, bits: int = 10
) -> Tuple[NDArray[np.uint32], NDArray[np.int32]]:
    """Return Morton codes and sorting indices for positions."""

    codes = morton_encode(positions, domain, bits=bits)
    order = np.argsort(codes, kind="stable")
    return codes, ensure_i32(order)
