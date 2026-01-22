import math
import cmath
import numpy as np
from scipy.special import factorial as fac
import pennylane as qml
from pennylane.ops import Identity
from pennylane import Device
from .._version import __version__
def squeezed_state(r, phi, hbar=2.0):
    """Returns a squeezed state.

    Args:
        r (float): the squeezing magnitude
        phi (float): the squeezing phase :math:`\\phi`
        hbar (float): (default 2) the value of :math:`\\hbar` in the commutation
            relation :math:`[\\x,\\p]=i\\hbar`

    Returns:
        array: the squeezed state
    """
    means = np.zeros(2)
    state = [squeezed_cov(r, phi, hbar), means]
    return state