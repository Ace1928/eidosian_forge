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
def create_trimat(self):
    """Create the full matrix `self.fullmat`, `self.d`, and `self.e`."""
    N = 10
    self.d = full(N, 1.0)
    self.e = full(N - 1, -1.0)
    self.full_mat = diag(self.d) + diag(self.e, -1) + diag(self.e, 1)
    ew, ev = linalg.eig(self.full_mat)
    ew = ew.real
    args = argsort(ew)
    self.w = ew[args]
    self.evec = ev[:, args]