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
def test_simple_fat_right(self):
    a = [[8, 2, 3], [2, 9, 5]]
    q, r = qr(a, mode='economic')
    c = [1, 2]
    cq, r2 = qr_multiply(a, c)
    assert_array_almost_equal(c @ q, cq)
    assert_array_almost_equal(r, r2)
    cq, r = qr_multiply(a, eye(2))
    assert_array_almost_equal(cq, q)