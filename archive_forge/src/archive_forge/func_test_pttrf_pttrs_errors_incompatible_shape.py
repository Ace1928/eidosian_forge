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
@pytest.mark.parametrize('ddtype,dtype', zip(REAL_DTYPES + REAL_DTYPES, DTYPES))
def test_pttrf_pttrs_errors_incompatible_shape(ddtype, dtype):
    n = 10
    pttrf = get_lapack_funcs('pttrf', dtype=dtype)
    d = generate_random_dtype_array((n,), ddtype) + 2
    e = generate_random_dtype_array((n - 1,), dtype)
    assert_raises(ValueError, pttrf, d[:-1], e)
    assert_raises(ValueError, pttrf, d, e[:-1])