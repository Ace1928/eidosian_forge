"""Metrics and direct reference implementations for validation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from algorithms_lab.core import Domain, ensure_f32


def direct_nbody_acceleration(
    positions: NDArray[np.float32],
    masses: NDArray[np.float32],
    domain: Domain,
    G: float = 1.0,
    softening: float = 1e-3,
) -> NDArray[np.float32]:
    """Compute exact N-body accelerations in O(N^2)."""

    pos = ensure_f32(positions)
    mass = np.asarray(masses, dtype=np.float32)
    delta = pos[:, None, :] - pos[None, :, :]
    delta = domain.minimal_image(delta)
    dist2 = np.einsum("ijk,ijk->ij", delta, delta) + softening * softening
    np.fill_diagonal(dist2, np.inf)
    dist = np.sqrt(dist2)
    inv = G * mass[None, :] / (dist2 * dist)
    acc = np.einsum("ij,ijk->ik", inv, -delta)
    return acc.astype(np.float32)


def l2_relative_error(reference: NDArray[np.float32], estimate: NDArray[np.float32]) -> float:
    """Return L2 relative error between arrays."""

    ref = np.asarray(reference, dtype=np.float32)
    est = np.asarray(estimate, dtype=np.float32)
    return float(np.linalg.norm(ref - est) / (np.linalg.norm(ref) + 1e-8))
