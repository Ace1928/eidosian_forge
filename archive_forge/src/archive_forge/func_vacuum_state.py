import math
import cmath
import numpy as np
from scipy.special import factorial as fac
import pennylane as qml
from pennylane.ops import Identity
from pennylane import Device
from .._version import __version__
def vacuum_state(wires, hbar=2.0):
    """Returns the vacuum state.

    Args:
        wires (int): the number of wires to initialize in the vacuum state
        hbar (float): (default 2) the value of :math:`\\hbar` in the commutation
            relation :math:`[\\x,\\p]=i\\hbar`
    Returns:
        array: the vacuum state
    """
    means = np.zeros(2 * wires)
    cov = np.identity(2 * wires) * hbar / 2
    state = [cov, means]
    return state