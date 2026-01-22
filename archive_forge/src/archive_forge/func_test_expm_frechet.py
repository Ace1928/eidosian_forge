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
def test_expm_frechet(self):
    M = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 1, 2], [0, 0, 5, 6]], dtype=float)
    A = np.array([[1, 2], [5, 6]], dtype=float)
    E = np.array([[3, 4], [7, 8]], dtype=float)
    expected_expm = scipy.linalg.expm(A)
    expected_frechet = scipy.linalg.expm(M)[:2, 2:]
    for kwargs in ({}, {'method': 'SPS'}, {'method': 'blockEnlarge'}):
        observed_expm, observed_frechet = expm_frechet(A, E, **kwargs)
        assert_allclose(expected_expm, observed_expm)
        assert_allclose(expected_frechet, observed_frechet)