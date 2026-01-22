import math
import cmath
import numpy as np
from scipy.special import factorial as fac
import pennylane as qml
from pennylane.ops import Identity
from pennylane import Device
from .._version import __version__
def fock_expectation(cov, mu, params, hbar=2.0):
    """Calculates the expectation and variance of a Fock state probability.

    Args:
        cov (array): :math:`2N\\times 2N` covariance matrix
        mu (array): length-:math:`2N` vector of means
        params (Sequence[int]): the Fock state to return the expectation value for
        hbar (float): (default 2) the value of :math:`\\hbar` in the commutation
            relation :math:`[\\x,\\p]=i\\hbar`

    Returns:
        tuple: the Fock state expectation and variance
    """
    ex = fock_prob(cov, mu, params[0], hbar=hbar)
    var = ex - ex ** 2
    return (ex, var)