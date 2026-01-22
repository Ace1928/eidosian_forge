import numpy as np  # type: ignore
from typing import Tuple, Optional
def set_Y_homog_rot_mtx(angle_rads: float, mtx: np.ndarray):
    """Update existing Y rotation matrix to new angle."""
    cosang = np.cos(angle_rads)
    sinang = np.sin(angle_rads)
    mtx[0][0] = mtx[2][2] = cosang
    mtx[0][2] = sinang
    mtx[2][0] = -sinang