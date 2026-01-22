import numpy as np  # type: ignore
from typing import Tuple, Optional
def coord_space(a0: np.ndarray, a1: np.ndarray, a2: np.ndarray, rev: bool=False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Generate transformation matrix to coordinate space defined by 3 points.

    New coordinate space will have:
        acs[0] on XZ plane
        acs[1] origin
        acs[2] on +Z axis

    :param NumPy column array x3 acs: X,Y,Z column input coordinates x3
    :param bool rev: if True, also return reverse transformation matrix
        (to return from coord_space)
    :returns: 4x4 NumPy array, x2 if rev=True
    """
    global gtm
    global gmry
    global gmrz, gmrz2
    tm = gtm
    mry = gmry
    mrz = gmrz
    mrz2 = gmrz2
    set_homog_trans_mtx(-a1[0], -a1[1], -a1[2], tm)
    p = a2 - a1
    sc = get_spherical_coordinates(p)
    set_Z_homog_rot_mtx(-sc[1], mrz)
    set_Y_homog_rot_mtx(-sc[2], mry)
    mt = gmry.dot(gmrz.dot(gtm))
    p = mt.dot(a0)
    azimuth2 = _get_azimuth(p[0], p[1])
    set_Z_homog_rot_mtx(-azimuth2, mrz2)
    mt = gmrz2.dot(mt)
    if not rev:
        return (mt, None)
    set_Z_homog_rot_mtx(azimuth2, mrz2)
    set_Y_homog_rot_mtx(sc[2], mry)
    set_Z_homog_rot_mtx(sc[1], mrz)
    set_homog_trans_mtx(a1[0], a1[1], a1[2], tm)
    mr = gtm.dot(gmrz.dot(gmry.dot(gmrz2)))
    return (mt, mr)