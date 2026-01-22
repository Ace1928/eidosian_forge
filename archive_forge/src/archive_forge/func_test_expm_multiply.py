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
def test_expm_multiply(self):
    np.random.seed(1234)
    n = 40
    k = 3
    nsamples = 10
    for i in range(nsamples):
        A = scipy.linalg.inv(np.random.randn(n, n))
        B = np.random.randn(n, k)
        observed = expm_multiply(A, B)
        expected = np.dot(sp_expm(A), B)
        assert_allclose(observed, expected)
        observed = estimated(expm_multiply)(aslinearoperator(A), B)
        assert_allclose(observed, expected)
        traceA = np.trace(A)
        observed = expm_multiply(aslinearoperator(A), B, traceA=traceA)
        assert_allclose(observed, expected)