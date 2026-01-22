from functools import partial
from itertools import product
import numpy as np
import pytest
from numpy.testing import (assert_allclose, assert_, assert_equal,
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.linalg import aslinearoperator
import scipy.linalg
from scipy.sparse.linalg import expm as sp_expm
from scipy.sparse.linalg._expm_multiply import (_theta, _compute_p_max,
from scipy._lib._util import np_long
def test_scaled_expm_multiply_single_timepoint(self):
    np.random.seed(1234)
    t = 0.1
    n = 5
    k = 2
    A = np.random.randn(n, n)
    B = np.random.randn(n, k)
    observed = _expm_multiply_simple(A, B, t=t)
    expected = sp_expm(t * A).dot(B)
    assert_allclose(observed, expected)
    observed = estimated(_expm_multiply_simple)(aslinearoperator(A), B, t=t)
    assert_allclose(observed, expected)