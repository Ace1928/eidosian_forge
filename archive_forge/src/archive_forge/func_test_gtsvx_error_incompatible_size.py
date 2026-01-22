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
@pytest.mark.parametrize('dtype', DTYPES * 2)
@pytest.mark.parametrize('trans_bool', [False, True])
@pytest.mark.parametrize('fact', ['F', 'N'])
def test_gtsvx_error_incompatible_size(dtype, trans_bool, fact):
    seed(42)
    gtsvx, gttrf = get_lapack_funcs(('gtsvx', 'gttrf'), dtype=dtype)
    n = 10
    dl = generate_random_dtype_array((n - 1,), dtype=dtype)
    d = generate_random_dtype_array((n,), dtype=dtype)
    du = generate_random_dtype_array((n - 1,), dtype=dtype)
    A = np.diag(dl, -1) + np.diag(d) + np.diag(du, 1)
    x = generate_random_dtype_array((n, 2), dtype=dtype)
    trans = 'T' if dtype in REAL_DTYPES else 'C'
    b = (A.conj().T if trans_bool else A) @ x
    dlf_, df_, duf_, du2f_, ipiv_, info_ = gttrf(dl, d, du) if fact == 'F' else [None] * 6
    if fact == 'N':
        assert_raises(ValueError, gtsvx, dl[:-1], d, du, b, fact=fact, trans=trans, dlf=dlf_, df=df_, duf=duf_, du2=du2f_, ipiv=ipiv_)
        assert_raises(ValueError, gtsvx, dl, d[:-1], du, b, fact=fact, trans=trans, dlf=dlf_, df=df_, duf=duf_, du2=du2f_, ipiv=ipiv_)
        assert_raises(ValueError, gtsvx, dl, d, du[:-1], b, fact=fact, trans=trans, dlf=dlf_, df=df_, duf=duf_, du2=du2f_, ipiv=ipiv_)
        assert_raises(Exception, gtsvx, dl, d, du, b[:-1], fact=fact, trans=trans, dlf=dlf_, df=df_, duf=duf_, du2=du2f_, ipiv=ipiv_)
    else:
        assert_raises(ValueError, gtsvx, dl, d, du, b, fact=fact, trans=trans, dlf=dlf_[:-1], df=df_, duf=duf_, du2=du2f_, ipiv=ipiv_)
        assert_raises(ValueError, gtsvx, dl, d, du, b, fact=fact, trans=trans, dlf=dlf_, df=df_[:-1], duf=duf_, du2=du2f_, ipiv=ipiv_)
        assert_raises(ValueError, gtsvx, dl, d, du, b, fact=fact, trans=trans, dlf=dlf_, df=df_, duf=duf_[:-1], du2=du2f_, ipiv=ipiv_)
        assert_raises(ValueError, gtsvx, dl, d, du, b, fact=fact, trans=trans, dlf=dlf_, df=df_, duf=duf_, du2=du2f_[:-1], ipiv=ipiv_)