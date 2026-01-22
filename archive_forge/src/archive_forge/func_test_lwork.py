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
def test_lwork(self):
    a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
    q, r = qr(a, lwork=None)
    q2, r2 = qr(a, lwork=3)
    assert_array_almost_equal(q2, q)
    assert_array_almost_equal(r2, r)
    q3, r3 = qr(a, lwork=10)
    assert_array_almost_equal(q3, q)
    assert_array_almost_equal(r3, r)
    q4, r4 = qr(a, lwork=-1)
    assert_array_almost_equal(q4, q)
    assert_array_almost_equal(r4, r)
    assert_raises(Exception, qr, (a,), {'lwork': 0})
    assert_raises(Exception, qr, (a,), {'lwork': 2})