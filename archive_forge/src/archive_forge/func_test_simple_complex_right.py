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
def test_simple_complex_right(self):
    a = [[3, 3 + 4j, 5], [5, 2, 2 + 7j], [3, 2, 7]]
    q, r = qr(a)
    c = [1, 2, 3 + 4j]
    qc, r = qr_multiply(a, c)
    assert_array_almost_equal(c @ q, qc)
    qc, r = qr_multiply(a, eye(3))
    assert_array_almost_equal(q, qc)