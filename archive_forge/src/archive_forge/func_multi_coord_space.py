import numpy as np  # type: ignore
from typing import Tuple, Optional
def multi_coord_space(a3: np.ndarray, dLen: int, rev: bool=False) -> np.ndarray:
    """Generate [dLen] transform matrices to coord space defined by 3 points.

    New coordinate space will have:
        acs[0] on XZ plane
        acs[1] origin
        acs[2] on +Z axis

    :param NumPy array [entries]x3x3 [entries] XYZ coords for 3 atoms
    :param bool rev: if True, also return reverse transformation matrix
    (to return from coord_space)
    :returns: [entries] 4x4 NumPy arrays, x2 if rev=True

    """
    tm = np.empty((dLen, 4, 4))
    tm[...] = np.identity(4)
    tm[:, 0:3, 3] = -a3[:, 1, 0:3]
    p = a3[:, 2] - a3[:, 1]
    r = np.linalg.norm(p, axis=1)
    azimuth = np.arctan2(p[:, 1], p[:, 0])
    polar_angle = np.arccos(np.divide(p[:, 2], r, where=r != 0))
    rz = multi_rot_Z(-azimuth)
    ry = multi_rot_Y(-polar_angle)
    mt = np.matmul(ry, np.matmul(rz, tm))
    p = np.matmul(mt, a3[:, 0].reshape(-1, 4, 1)).reshape(-1, 4)
    azimuth2 = np.arctan2(p[:, 1], p[:, 0])
    rz2 = multi_rot_Z(-azimuth2)
    if not rev:
        return np.matmul(rz2, mt[:])
    mt = np.matmul(rz2, mt[:])
    mrz2 = multi_rot_Z(azimuth2)
    mry = multi_rot_Y(polar_angle)
    mrz = multi_rot_Z(azimuth)
    tm[:, 0:3, 3] = a3[:, 1, 0:3]
    mr = tm @ mrz @ mry @ mrz2
    return np.array([mt, mr])