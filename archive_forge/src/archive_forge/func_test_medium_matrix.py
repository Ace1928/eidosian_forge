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
@pytest.mark.slow
@pytest.mark.skip(reason='this test is deliberately slow')
def test_medium_matrix(self):
    n = 1000
    A = np.random.exponential(size=(n, n))
    E = np.random.exponential(size=(n, n))
    sps_expm, sps_frechet = expm_frechet(A, E, method='SPS')
    blockEnlarge_expm, blockEnlarge_frechet = expm_frechet(A, E, method='blockEnlarge')
    assert_allclose(sps_expm, blockEnlarge_expm)
    assert_allclose(sps_frechet, blockEnlarge_frechet)