from __future__ import annotations
import math
import numpy as np
def to_zyz(self) -> np.ndarray:
    """Converts a unit-length quaternion to a sequence
        of ZYZ Euler angles.

        Returns:
            ndarray: Array of Euler angles.
        """
    mat = self.to_matrix()
    euler = np.zeros(3, dtype=float)
    if mat[2, 2] < 1:
        if mat[2, 2] > -1:
            euler[0] = math.atan2(mat[1, 2], mat[0, 2])
            euler[1] = math.acos(mat[2, 2])
            euler[2] = math.atan2(mat[2, 1], -mat[2, 0])
        else:
            euler[0] = -math.atan2(mat[1, 0], mat[1, 1])
            euler[1] = np.pi
    else:
        euler[0] = math.atan2(mat[1, 0], mat[1, 1])
    return euler