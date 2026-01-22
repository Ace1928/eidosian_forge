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
def test_round_trip_random_complex(self):
    np.random.seed(1234)
    for p in range(1, 5):
        for n in range(1, 5):
            M_unscaled = np.random.randn(n, n) + 1j * np.random.randn(n, n)
            for scale in np.logspace(-4, 4, 9):
                M = M_unscaled * scale
                M_root = fractional_matrix_power(M, 1 / p)
                M_round_trip = np.linalg.matrix_power(M_root, p)
                assert_allclose(M_round_trip, M)