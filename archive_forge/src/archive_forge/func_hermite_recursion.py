import numpy as np
from numpy import array, sqrt
from numpy.testing import (assert_array_almost_equal, assert_equal,
from pytest import raises as assert_raises
from scipy import integrate
import scipy.special as sc
from scipy.special import gamma
import scipy.special._orthogonal as orth
def hermite_recursion(n, nodes):
    H = np.zeros((n, nodes.size))
    H[0, :] = np.pi ** (-0.25) * np.exp(-0.5 * nodes ** 2)
    if n > 1:
        H[1, :] = sqrt(2.0) * nodes * H[0, :]
        for k in range(2, n):
            H[k, :] = sqrt(2.0 / k) * nodes * H[k - 1, :] - sqrt((k - 1.0) / k) * H[k - 2, :]
    return H