import math
import cmath
import numpy as np
from scipy.special import factorial as fac
import pennylane as qml
from pennylane.ops import Identity
from pennylane import Device
from .._version import __version__
def displaced_squeezed_state(a, phi_a, r, phi_r, hbar=2.0):
    """Returns a squeezed coherent state

    Args:
        a (real): the displacement magnitude
        phi_a (real): the displacement phase
        r (float): the squeezing magnitude
        phi_r (float): the squeezing phase :math:`\\phi_r`
        hbar (float): (default 2) the value of :math:`\\hbar` in the commutation
            relation :math:`[\\x,\\p]=i\\hbar`

    Returns:
        array: the squeezed coherent state
    """
    alpha = a * cmath.exp(1j * phi_a)
    means = np.array([alpha.real, alpha.imag]) * math.sqrt(2 * hbar)
    state = [squeezed_cov(r, phi_r, hbar), means]
    return state