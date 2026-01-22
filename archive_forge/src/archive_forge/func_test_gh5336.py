import random
import functools
import numpy as np
from numpy import array, identity, dot, sqrt
from numpy.testing import (assert_array_almost_equal, assert_allclose, assert_,
import pytest
import scipy.linalg
from scipy.linalg import (funm, signm, logm, sqrtm, fractional_matrix_power,
from scipy.linalg import _matfuncs_inv_ssq
import scipy.linalg._expm_frechet
from scipy.optimize import minimize
def test_gh5336(self):
    M = np.diag([2, 1, 0])
    R = np.diag([sqrt(2), 1, 0])
    assert_allclose(np.dot(R, R), M, atol=1e-14)
    assert_allclose(sqrtm(M), R, atol=1e-14)