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
def test_bad(self):
    e = 2 ** (-5)
    se = sqrt(e)
    a = array([[1.0, 0, 0, 1], [0, e, 0, 0], [0, 0, e, 0], [0, 0, 0, 1]])
    sa = array([[1, 0, 0, 0.5], [0, se, 0, 0], [0, 0, se, 0], [0, 0, 0, 1]])
    n = a.shape[0]
    assert_array_almost_equal(dot(sa, sa), a)
    esa = sqrtm(a, disp=False, blocksize=n)[0]
    assert_array_almost_equal(dot(esa, esa), a)
    esa = sqrtm(a, disp=False, blocksize=2)[0]
    assert_array_almost_equal(dot(esa, esa), a)