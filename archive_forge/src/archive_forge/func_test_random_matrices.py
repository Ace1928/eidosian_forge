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
@pytest.mark.parametrize('dtype,trans', [(dtype, trans) for dtype in DTYPES for trans in ['N', 'T', 'C'] if not (trans == 'C' and dtype in REAL_DTYPES)])
@pytest.mark.parametrize('uplo', ['U', 'L'])
@pytest.mark.parametrize('diag', ['N', 'U'])
def test_random_matrices(self, dtype, trans, uplo, diag):
    seed(1724)
    n, nrhs, kd = (4, 3, 2)
    tbtrs = get_lapack_funcs('tbtrs', dtype=dtype)
    is_upper = uplo == 'U'
    ku = kd * is_upper
    kl = kd - ku
    band_offsets = range(ku, -kl - 1, -1)
    band_widths = [n - abs(x) for x in band_offsets]
    bands = [generate_random_dtype_array((width,), dtype) for width in band_widths]
    if diag == 'U':
        bands[ku] = np.ones(n, dtype=dtype)
    a = sps.diags(bands, band_offsets, format='dia')
    ab = np.zeros((kd + 1, n), dtype)
    for row, k in enumerate(band_offsets):
        ab[row, max(k, 0):min(n + k, n)] = a.diagonal(k)
    b = generate_random_dtype_array((n, nrhs), dtype)
    x, info = tbtrs(ab=ab, b=b, uplo=uplo, trans=trans, diag=diag)
    assert_equal(info, 0)
    if trans == 'N':
        assert_allclose(a @ x, b, rtol=5e-05)
    elif trans == 'T':
        assert_allclose(a.T @ x, b, rtol=5e-05)
    elif trans == 'C':
        assert_allclose(a.H @ x, b, rtol=5e-05)
    else:
        raise ValueError('Invalid trans argument')