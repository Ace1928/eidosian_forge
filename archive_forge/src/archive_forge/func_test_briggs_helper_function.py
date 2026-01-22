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
def test_briggs_helper_function(self):
    np.random.seed(1234)
    for a in np.random.randn(10) + 1j * np.random.randn(10):
        for k in range(5):
            x_observed = _matfuncs_inv_ssq._briggs_helper_function(a, k)
            x_expected = a ** np.exp2(-k) - 1
            assert_allclose(x_observed, x_expected)