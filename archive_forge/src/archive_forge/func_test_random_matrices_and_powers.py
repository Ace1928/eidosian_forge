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
def test_random_matrices_and_powers(self):
    np.random.seed(1234)
    nsamples = 20
    for i in range(nsamples):
        n = random.randrange(1, 5)
        p = np.random.randn()
        matrix_scale = np.exp(random.randrange(-4, 5))
        A = np.random.randn(n, n)
        if random.choice((True, False)):
            A = A + 1j * np.random.randn(n, n)
        A = A * matrix_scale
        A_power = fractional_matrix_power(A, p)
        A_logm, info = logm(A, disp=False)
        A_power_expm_logm = expm(A_logm * p)
        assert_allclose(A_power, A_power_expm_logm)