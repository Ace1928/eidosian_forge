import math
import cmath
import numpy as np
from scipy.special import factorial as fac
import pennylane as qml
from pennylane.ops import Identity
from pennylane import Device
from .._version import __version__
def gaussian_state(cov, mu, hbar=2.0):
    """Returns a Gaussian state.

    This is simply a bare wrapper function,
    since the covariance matrix and means vector
    can be passed via the parameters unchanged.

    Note that both the covariance and means vector
    matrix should be in :math:`(\\x_1,\\dots, \\x_N, \\p_1, \\dots, \\p_N)`
    ordering.

    Args:
        cov (array): covariance matrix. Must be dimension :math:`2N\\times 2N`,
            where N is the number of modes
        mu (array): vector means. Must be length-:math:`2N`,
            where N is the number of modes
        hbar (float): (default 2) the value of :math:`\\hbar` in the commutation
            relation :math:`[\\x,\\p]=i\\hbar`

    Returns:
        tuple: the mean and the covariance matrix of the Gaussian state
    """
    return (cov, mu)