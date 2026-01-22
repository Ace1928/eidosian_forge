import sys
from math import sin, cos, acos, sqrt
import numpy as np
from qiskit.exceptions import MissingOptionalLibraryError
@staticmethod
def from_axisangle(theta, v):
    """Create quaternion from axis"""
    v = _normalize(v)
    new_quaternion = _Quaternion()
    new_quaternion._axisangle_to_q(theta, v)
    return new_quaternion