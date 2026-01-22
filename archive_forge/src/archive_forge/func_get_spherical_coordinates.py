import numpy as np  # type: ignore
from typing import Tuple, Optional
def get_spherical_coordinates(xyz: np.array) -> Tuple[float, float, float]:
    """Compute spherical coordinates (r, azimuth, polar_angle) for X,Y,Z point.

    :param array xyz: column vector (3 row x 1 column NumPy array)
    :return: tuple of r, azimuth, polar_angle for input coordinate
    """
    r = np.linalg.norm(xyz)
    if 0 == r:
        return (0, 0, 0)
    azimuth = _get_azimuth(xyz[0], xyz[1])
    polar_angle = np.arccos(xyz[2] / r)
    return (r, azimuth, polar_angle)