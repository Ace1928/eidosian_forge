import math
import cmath
import numpy as np
from scipy.special import factorial as fac
import pennylane as qml
from pennylane.ops import Identity
from pennylane import Device
from .._version import __version__
def controlled_addition(s):
    """CX gate.

    Args:
        s (float): gate parameter

    Returns:
        array: symplectic transformation matrix
    """
    S = np.array([[1, 0, 0, 0], [s, 1, 0, 0], [0, 0, 1, -s], [0, 0, 0, 1]])
    return S