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
@pytest.mark.parametrize('dtype_', DTYPES)
@pytest.mark.parametrize('m', [1, 10, 100, 1000])
def test_orcsd_uncsd_lwork(dtype_, m):
    seed(1234)
    p = randint(0, m)
    q = m - p
    pfx = 'or' if dtype_ in REAL_DTYPES else 'un'
    dlw = pfx + 'csd_lwork'
    lw = get_lapack_funcs(dlw, dtype=dtype_)
    lwval = _compute_lwork(lw, m, p, q)
    lwval = lwval if pfx == 'un' else (lwval,)
    assert all([x > 0 for x in lwval])