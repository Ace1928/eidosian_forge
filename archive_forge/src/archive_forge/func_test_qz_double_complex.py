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
def test_qz_double_complex(self):
    rng = np.random.RandomState(12345)
    n = 5
    A = rng.random([n, n])
    B = rng.random([n, n])
    AA, BB, Q, Z = qz(A, B, output='complex')
    aa = Q @ AA @ Z.conj().T
    assert_array_almost_equal(aa.real, A)
    assert_array_almost_equal(aa.imag, 0)
    bb = Q @ BB @ Z.conj().T
    assert_array_almost_equal(bb.real, B)
    assert_array_almost_equal(bb.imag, 0)
    assert_array_almost_equal(Q @ Q.conj().T, eye(n))
    assert_array_almost_equal(Z @ Z.conj().T, eye(n))
    assert_(np.all(diag(BB) >= 0))