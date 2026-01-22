import numpy as np
from ase.atoms import Atoms
@staticmethod
def rotate_byq(q, vector):
    """Apply the rotation matrix to a vector."""
    qw, qx, qy, qz = (q[0], q[1], q[2], q[3])
    x, y, z = (vector[0], vector[1], vector[2])
    ww = qw * qw
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    return np.array([(ww + xx - yy - zz) * x + 2 * ((xy - wz) * y + (xz + wy) * z), (ww - xx + yy - zz) * y + 2 * ((xy + wz) * x + (yz - wx) * z), (ww - xx - yy + zz) * z + 2 * ((xz - wy) * x + (yz + wx) * y)])