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
def test_dgbtrs(self):
    """Compare dgbtrs  solutions for linear equation system  A*x = b
           with solutions of linalg.solve."""
    lu_symm_band, ipiv, info = dgbtrf(self.bandmat_real, self.KL, self.KU)
    y, info = dgbtrs(lu_symm_band, self.KL, self.KU, self.b, ipiv)
    y_lin = linalg.solve(self.real_mat, self.b)
    assert_array_almost_equal(y, y_lin)