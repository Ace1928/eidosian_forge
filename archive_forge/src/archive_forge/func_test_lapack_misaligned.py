import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (eig, eigvals, lu, svd, svdvals, cholesky, qr,
from scipy.linalg.lapack import (dgbtrf, dgbtrs, zgbtrf, zgbtrs, dsbev,
from scipy.linalg._misc import norm
from scipy.linalg._decomp_qz import _select_function
from scipy.stats import ortho_group
from numpy import (array, diag, full, linalg, argsort, zeros, arange,
from scipy.linalg._testutils import assert_no_overwrite
from scipy.sparse._sputils import matrix
from scipy._lib._testutils import check_free_memory
from scipy.linalg.blas import HAS_ILP64
@pytest.mark.xfail(run=False, reason='Ticket #1152, triggers a segfault in rare cases.')
def test_lapack_misaligned():
    M = np.eye(10, dtype=float)
    R = np.arange(100)
    R.shape = (10, 10)
    S = np.arange(20000, dtype=np.uint8)
    S = np.frombuffer(S.data, offset=4, count=100, dtype=float)
    S.shape = (10, 10)
    b = np.ones(10)
    LU, piv = lu_factor(S)
    for func, args, kwargs in [(eig, (S,), dict(overwrite_a=True)), (eigvals, (S,), dict(overwrite_a=True)), (lu, (S,), dict(overwrite_a=True)), (lu_factor, (S,), dict(overwrite_a=True)), (lu_solve, ((LU, piv), b), dict(overwrite_b=True)), (solve, (S, b), dict(overwrite_a=True, overwrite_b=True)), (svd, (M,), dict(overwrite_a=True)), (svd, (R,), dict(overwrite_a=True)), (svd, (S,), dict(overwrite_a=True)), (svdvals, (S,), dict()), (svdvals, (S,), dict(overwrite_a=True)), (cholesky, (M,), dict(overwrite_a=True)), (qr, (S,), dict(overwrite_a=True)), (rq, (S,), dict(overwrite_a=True)), (hessenberg, (S,), dict(overwrite_a=True)), (schur, (S,), dict(overwrite_a=True))]:
        check_lapack_misaligned(func, args, kwargs)