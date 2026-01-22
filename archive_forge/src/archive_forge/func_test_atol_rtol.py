import itertools
import warnings
import numpy as np
from numpy import (arange, array, dot, zeros, identity, conjugate, transpose,
from numpy.random import random
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (solve, inv, det, lstsq, pinv, pinvh, norm,
from scipy.linalg._testutils import assert_no_overwrite
from scipy._lib._testutils import check_free_memory, IS_MUSL
from scipy.linalg.blas import HAS_ILP64
from scipy._lib.deprecation import _NoValue
def test_atol_rtol(self):
    n = 12
    q, _ = qr(np.random.rand(n, n))
    a = np.diag([4, 3, 2, 1, 9.9e-05, 9.9e-06] + [9.9e-07] * (n - 6))
    a = q.T @ a @ q
    a_m = np.diag([4, 3, 2, 1, 9.9e-05, 0.0] + [0.0] * (n - 6))
    a_m = q.T @ a_m @ q
    atol = 1e-05
    rtol = (0.000401 - 4e-05) / 4
    a_p = pinvh(a, atol=atol, rtol=0.0)
    adiff1 = a @ a_p @ a - a
    adiff2 = a_m @ a_p @ a_m - a_m
    assert_allclose(norm(adiff1), atol, rtol=0.1)
    assert_allclose(norm(adiff2), 1e-12, atol=1e-11)
    a_p = pinvh(a, atol=atol, rtol=rtol)
    adiff1 = a @ a_p @ a - a
    adiff2 = a_m @ a_p @ a_m - a_m
    assert_allclose(norm(adiff1), 0.0001, rtol=0.1)
    assert_allclose(norm(adiff2), 0.0001, rtol=0.1)