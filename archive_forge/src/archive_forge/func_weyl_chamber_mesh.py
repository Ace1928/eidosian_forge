import itertools
from typing import Union, Sequence, Optional
import numpy as np
from cirq.value import random_state
def weyl_chamber_mesh(spacing: float) -> np.ndarray:
    """Cubic mesh of points in the Weyl chamber.

    Args:
        spacing: Euclidean distance between neighboring KAK vectors.

    Returns:
        np.ndarray of shape (N,3) corresponding to the points in the Weyl
        chamber.

    Raises:
        ValueError: If the spacing is so small (less than 1e-3) that this
            would build a mesh of size about 1GB.
    """
    if spacing < 0.001:
        raise ValueError(f'Generating a mesh with spacing {spacing} may cause system to crash.')
    disps = np.arange(-np.pi / 4, np.pi / 4, step=spacing)
    mesh_points = np.array([a.ravel() for a in np.array(np.meshgrid(*(disps,) * 3))])
    mesh_points = np.moveaxis(mesh_points, 0, -1)
    return mesh_points[in_weyl_chamber(mesh_points)]