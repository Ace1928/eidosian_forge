import math
import cmath
import numpy as np
from scipy.special import factorial as fac
import pennylane as qml
from pennylane.ops import Identity
from pennylane import Device
from .._version import __version__
def two_mode_squeezing(r, phi):
    """Two-mode squeezing.

    Args:
        r (float): squeezing magnitude
        phi (float): rotation parameter

    Returns:
        array: symplectic transformation matrix
    """
    cp = math.cos(phi)
    sp = math.sin(phi)
    ch = math.cosh(r)
    sh = math.sinh(r)
    S = np.array([[ch, cp * sh, 0, sp * sh], [cp * sh, ch, sp * sh, 0], [0, sp * sh, ch, -cp * sh], [sp * sh, 0, -cp * sh, ch]])
    return S