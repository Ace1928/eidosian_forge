import sys
from functools import reduce
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import (eye, ones, zeros, zeros_like, triu, tril, tril_indices,
from numpy.random import rand, randint, seed
from scipy.linalg import (_flapack as flapack, lapack, inv, svd, cholesky,
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
import scipy.sparse as sps
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.blas import get_blas_funcs
@pytest.mark.skipif(True, reason='Failing on some OpenBLAS version, see gh-12276')
def test_heequb():
    A = np.diag([2] * 5 + [1002] * 5) + np.diag(np.ones(9), k=1) * 1j
    s, scond, amax, info = lapack.zheequb(A)
    assert_equal(info, 0)
    assert_allclose(np.log2(s), [0.0, -1.0] * 2 + [0.0] + [-4] * 5)
    A = np.diag(2 ** np.abs(np.arange(-5, 6)) + 0j)
    A[5, 5] = 1024
    A[5, 0] = 16j
    s, scond, amax, info = lapack.cheequb(A.astype(np.complex64), lower=1)
    assert_equal(info, 0)
    assert_allclose(np.log2(s), [-2, -1, -1, 0, 0, -5, 0, -1, -1, -2, -2])