"""SPH kernel functions for 2D and 3D."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def poly6(r: NDArray[np.float32], h: float, dims: int) -> NDArray[np.float32]:
    """Poly6 smoothing kernel."""

    if dims == 2:
        c = 4.0 / (np.pi * h**8)
    elif dims == 3:
        c = 315.0 / (64.0 * np.pi * h**9)
    else:
        raise ValueError("dims must be 2 or 3")
    diff = h * h - r * r
    diff = np.maximum(diff, 0.0)
    return (c * diff**3).astype(np.float32)


def spiky_grad(
    delta: NDArray[np.float32], r: NDArray[np.float32], h: float, dims: int
) -> NDArray[np.float32]:
    """Gradient of spiky kernel."""

    if dims == 2:
        c = -30.0 / (np.pi * h**5)
    elif dims == 3:
        c = -45.0 / (np.pi * h**6)
    else:
        raise ValueError("dims must be 2 or 3")
    diff = h - r
    diff = np.maximum(diff, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = c * diff * diff / np.maximum(r, 1e-6)
    return delta * scale[:, None]


def viscosity_laplacian(r: NDArray[np.float32], h: float, dims: int) -> NDArray[np.float32]:
    """Laplacian of viscosity kernel."""

    if dims == 2:
        c = 40.0 / (np.pi * h**5)
    elif dims == 3:
        c = 45.0 / (np.pi * h**6)
    else:
        raise ValueError("dims must be 2 or 3")
    diff = h - r
    diff = np.maximum(diff, 0.0)
    return (c * diff).astype(np.float32)
