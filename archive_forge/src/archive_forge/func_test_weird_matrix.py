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
def test_weird_matrix(self):
    for dt in (int, float):
        A = np.array([[0, 0, 1], [0, 0, 0], [0, 1, 0]], dtype=dt)
        B = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=dt)
        assert_array_equal(B, A.dot(A))
        B_sqrtm, info = sqrtm(B, disp=False)
        assert_(np.isnan(B_sqrtm).all())