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
def test_sort_explicit(self):
    A1 = np.eye(2)
    B1 = np.diag([-2, 0.5])
    expected1 = [('lhp', [-0.5, 2]), ('rhp', [2, -0.5]), ('iuc', [-0.5, 2]), ('ouc', [2, -0.5])]
    A2 = np.eye(2)
    B2 = np.diag([-2 + 1j, 0.5 + 0.5j])
    expected2 = [('lhp', [1 / (-2 + 1j), 1 / (0.5 + 0.5j)]), ('rhp', [1 / (0.5 + 0.5j), 1 / (-2 + 1j)]), ('iuc', [1 / (-2 + 1j), 1 / (0.5 + 0.5j)]), ('ouc', [1 / (0.5 + 0.5j), 1 / (-2 + 1j)])]
    A3 = np.eye(2)
    B3 = np.diag([2, 0])
    expected3 = [('rhp', [0.5, np.inf]), ('iuc', [0.5, np.inf]), ('ouc', [np.inf, 0.5])]
    A4 = np.eye(2)
    B4 = np.diag([-2, 0])
    expected4 = [('lhp', [-0.5, np.inf]), ('iuc', [-0.5, np.inf]), ('ouc', [np.inf, -0.5])]
    A5 = np.diag([0, 1])
    B5 = np.diag([0, 0.5])
    expected5 = [('rhp', [2, np.nan]), ('ouc', [2, np.nan])]
    A = [A1, A2, A3, A4, A5]
    B = [B1, B2, B3, B4, B5]
    expected = [expected1, expected2, expected3, expected4, expected5]
    for Ai, Bi, expectedi in zip(A, B, expected):
        for sortstr, expected_eigvals in expectedi:
            _, _, alpha, beta, _, _ = ordqz(Ai, Bi, sort=sortstr)
            azero = alpha == 0
            bzero = beta == 0
            x = np.empty_like(alpha)
            x[azero & bzero] = np.nan
            x[~azero & bzero] = np.inf
            x[~bzero] = alpha[~bzero] / beta[~bzero]
            assert_allclose(expected_eigvals, x)