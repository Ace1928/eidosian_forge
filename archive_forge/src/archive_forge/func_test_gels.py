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
def test_gels(self):
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        m = 10
        n = 20
        nrhs = 1
        a1 = rand(m, n).astype(dtype)
        b1 = rand(n).astype(dtype)
        gls, glslw = get_lapack_funcs(('gels', 'gels_lwork'), dtype=dtype)
        lwork = _compute_lwork(glslw, m, n, nrhs)
        _, _, info = gls(a1, b1, lwork=lwork)
        assert_(info >= 0)
        _, _, info = gls(a1, b1, trans='TTCC'[ind], lwork=lwork)
        assert_(info >= 0)
    for dtype in REAL_DTYPES:
        a1 = np.array([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]], dtype=dtype)
        b1 = np.array([16.0, 17.0, 20.0], dtype=dtype)
        gels, gels_lwork, geqrf = get_lapack_funcs(('gels', 'gels_lwork', 'geqrf'), (a1, b1))
        m, n = a1.shape
        if len(b1.shape) == 2:
            nrhs = b1.shape[1]
        else:
            nrhs = 1
        lwork = _compute_lwork(gels_lwork, m, n, nrhs)
        lqr, x, info = gels(a1, b1, lwork=lwork)
        assert_allclose(x[:-1], np.array([-14.333333333333323, 14.999999999999991], dtype=dtype), rtol=25 * np.finfo(dtype).eps)
        lqr_truth, _, _, _ = geqrf(a1)
        assert_array_equal(lqr, lqr_truth)
    for dtype in COMPLEX_DTYPES:
        a1 = np.array([[1.0 + 4j, 2.0], [4.0 + 0.5j, 5.0 - 3j], [7.0 - 2j, 8.0 + 0.7j]], dtype=dtype)
        b1 = np.array([16.0, 17.0 + 2j, 20.0 - 4j], dtype=dtype)
        gels, gels_lwork, geqrf = get_lapack_funcs(('gels', 'gels_lwork', 'geqrf'), (a1, b1))
        m, n = a1.shape
        if len(b1.shape) == 2:
            nrhs = b1.shape[1]
        else:
            nrhs = 1
        lwork = _compute_lwork(gels_lwork, m, n, nrhs)
        lqr, x, info = gels(a1, b1, lwork=lwork)
        assert_allclose(x[:-1], np.array([1.161753632288328 - 1.901075709391912j, 1.735882340522193 + 1.521240901196909j], dtype=dtype), rtol=25 * np.finfo(dtype).eps)
        lqr_truth, _, _, _ = geqrf(a1)
        assert_array_equal(lqr, lqr_truth)