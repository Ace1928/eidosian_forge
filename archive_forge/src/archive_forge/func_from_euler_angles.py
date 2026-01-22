import numpy as np
from ase.atoms import Atoms
@staticmethod
def from_euler_angles(a, b, c, mode='zyz'):
    """Build quaternion from Euler angles, given in radians. Default
        mode is ZYZ, but it can be set to ZXZ as well."""
    q_a = Quaternion.from_axis_angle([0, 0, 1], a)
    q_c = Quaternion.from_axis_angle([0, 0, 1], c)
    if mode == 'zyz':
        q_b = Quaternion.from_axis_angle([0, 1, 0], b)
    elif mode == 'zxz':
        q_b = Quaternion.from_axis_angle([1, 0, 0], b)
    else:
        raise ValueError('Invalid Euler angles mode {0}'.format(mode))
    return q_c * q_b * q_a