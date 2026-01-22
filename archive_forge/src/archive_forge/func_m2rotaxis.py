import numpy as np  # type: ignore
from typing import Tuple, Optional
def m2rotaxis(m):
    """Return angles, axis pair that corresponds to rotation matrix m.

    The case where ``m`` is the identity matrix corresponds to a singularity
    where any rotation axis is valid. In that case, ``Vector([1, 0, 0])``,
    is returned.
    """
    eps = 1e-05
    if abs(m[0, 1] - m[1, 0]) < eps and abs(m[0, 2] - m[2, 0]) < eps and (abs(m[1, 2] - m[2, 1]) < eps):
        if abs(m[0, 1] + m[1, 0]) < eps and abs(m[0, 2] + m[2, 0]) < eps and (abs(m[1, 2] + m[2, 1]) < eps) and (abs(m[0, 0] + m[1, 1] + m[2, 2] - 3) < eps):
            angle = 0
        else:
            angle = np.pi
    else:
        t = 0.5 * (np.trace(m) - 1)
        t = max(-1, t)
        t = min(1, t)
        angle = np.arccos(t)
    if angle < 1e-15:
        return (0.0, Vector(1, 0, 0))
    elif angle < np.pi:
        x = m[2, 1] - m[1, 2]
        y = m[0, 2] - m[2, 0]
        z = m[1, 0] - m[0, 1]
        axis = Vector(x, y, z)
        axis.normalize()
        return (angle, axis)
    else:
        m00 = m[0, 0]
        m11 = m[1, 1]
        m22 = m[2, 2]
        if m00 > m11 and m00 > m22:
            x = np.sqrt(m00 - m11 - m22 + 0.5)
            y = m[0, 1] / (2 * x)
            z = m[0, 2] / (2 * x)
        elif m11 > m00 and m11 > m22:
            y = np.sqrt(m11 - m00 - m22 + 0.5)
            x = m[0, 1] / (2 * y)
            z = m[1, 2] / (2 * y)
        else:
            z = np.sqrt(m22 - m00 - m11 + 0.5)
            x = m[0, 2] / (2 * z)
            y = m[1, 2] / (2 * z)
        axis = Vector(x, y, z)
        axis.normalize()
        return (np.pi, axis)