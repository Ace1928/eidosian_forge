import itertools
import platform
import sys
import pytest
import numpy as np
from numpy import ones, r_, diag
from numpy.testing import (assert_almost_equal, assert_equal,
from scipy import sparse
from scipy.linalg import eig, eigh, toeplitz, orth
from scipy.sparse import spdiags, diags, eye, csr_matrix
from scipy.sparse.linalg import eigs, LinearOperator
from scipy.sparse.linalg._eigen.lobpcg import lobpcg
from scipy.sparse.linalg._eigen.lobpcg.lobpcg import _b_orthonormalize
from scipy._lib._util import np_long, np_ulong
@pytest.mark.xfail(_IS_32BIT and sys.platform == 'win32', reason='tolerance violation on windows')
@pytest.mark.xfail(platform.machine() == 'ppc64le', reason='fails on ppc64le')
@pytest.mark.filterwarnings('ignore:Exited postprocessing')
def test_tolerance_float32():
    """Check lobpcg for attainable tolerance in float32.
    """
    rnd = np.random.RandomState(0)
    n = 50
    m = 3
    vals = -np.arange(1, n + 1)
    A = diags([vals], [0], (n, n))
    A = A.astype(np.float32)
    X = rnd.standard_normal((n, m))
    X = X.astype(np.float32)
    eigvals, _ = lobpcg(A, X, tol=1.25e-05, maxiter=50, verbosityLevel=0)
    assert_allclose(eigvals, -np.arange(1, 1 + m), atol=2e-05, rtol=1e-05)