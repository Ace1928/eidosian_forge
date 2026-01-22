import sys
import numpy as np
from numpy.testing import (assert_, assert_array_equal, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
def shock_sol(x):
    eps = 0.001
    k = np.sqrt(2 * eps)
    return np.cos(np.pi * x) + erf(x / k) / erf(1 / k)