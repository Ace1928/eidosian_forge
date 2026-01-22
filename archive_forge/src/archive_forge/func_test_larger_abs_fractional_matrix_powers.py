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
def test_larger_abs_fractional_matrix_powers(self):
    np.random.seed(1234)
    for n in (2, 3, 5):
        for i in range(10):
            M = np.random.randn(n, n) + 1j * np.random.randn(n, n)
            M_one_fifth = fractional_matrix_power(M, 0.2)
            M_round_trip = np.linalg.matrix_power(M_one_fifth, 5)
            assert_allclose(M, M_round_trip)
            X = fractional_matrix_power(M, -5.4)
            Y = np.linalg.matrix_power(M_one_fifth, -27)
            assert_allclose(X, Y)
            X = fractional_matrix_power(M, 3.8)
            Y = np.linalg.matrix_power(M_one_fifth, 19)
            assert_allclose(X, Y)