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
@pytest.mark.parametrize('driver', ['ev', 'evd', 'evr', 'evx'])
@pytest.mark.parametrize('pfx', ['sy', 'he'])
def test_standard_eigh_lworks(pfx, driver):
    n = 1200
    dtype = REAL_DTYPES if pfx == 'sy' else COMPLEX_DTYPES
    sc_dlw = get_lapack_funcs(pfx + driver + '_lwork', dtype=dtype[0])
    dz_dlw = get_lapack_funcs(pfx + driver + '_lwork', dtype=dtype[1])
    try:
        _compute_lwork(sc_dlw, n, lower=1)
        _compute_lwork(dz_dlw, n, lower=1)
    except Exception as e:
        pytest.fail(f'{pfx + driver}_lwork raised unexpected exception: {e}')