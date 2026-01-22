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
@pytest.mark.slow
def test_decompose_ouc(self):
    rng = np.random.RandomState(12345)
    N = 202
    for ddtype in [np.float32, np.float64, np.complex128, np.complex64]:
        A = rng.random((N, N)).astype(ddtype)
        B = rng.random((N, N)).astype(ddtype)
        S, T, alpha, beta, U, V = ordqz(A, B, sort='ouc')