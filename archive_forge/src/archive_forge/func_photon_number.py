import math
import cmath
import numpy as np
from scipy.special import factorial as fac
import pennylane as qml
from pennylane.ops import Identity
from pennylane import Device
from .._version import __version__
def photon_number(cov, mu, params, hbar=2.0):
    """Calculates the mean photon number for a given one-mode state.

    Args:
        cov (array): :math:`2\\times 2` covariance matrix
        mu (array): length-2 vector of means
        params (None): no parameters are used for this expectation value
        hbar (float): (default 2) the value of :math:`\\hbar` in the commutation
            relation :math:`[\\x,\\p]=i\\hbar`

    Returns:
        tuple: contains the photon number expectation and variance
    """
    ex = (np.trace(cov) + mu.T @ mu) / (2 * hbar) - 1 / 2
    var = (np.trace(cov @ cov) + 2 * mu.T @ cov @ mu) / (2 * hbar ** 2) - 1 / 4
    return (ex, var)