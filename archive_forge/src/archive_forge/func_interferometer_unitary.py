import math
import cmath
import numpy as np
from scipy.special import factorial as fac
import pennylane as qml
from pennylane.ops import Identity
from pennylane import Device
from .._version import __version__
def interferometer_unitary(U):
    """InterferometerUnitary

    Args:
        U (array): unitary matrix

    Returns:
        array: symplectic transformation matrix
    """
    N = 2 * len(U)
    X = U.real
    Y = U.imag
    rows = np.arange(N).reshape(2, -1).T.flatten()
    S = np.vstack([np.hstack([X, -Y]), np.hstack([Y, X])])[:, rows][rows]
    return S