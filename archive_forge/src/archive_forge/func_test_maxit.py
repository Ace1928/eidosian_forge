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
def test_maxit():
    """Check lobpcg if maxit=maxiter runs maxiter iterations and
    if maxit=None runs 20 iterations (the default)
    by checking the size of the iteration history output, which should
    be the number of iterations plus 3 (initial, final, and postprocessing)
    typically when maxiter is small and the choice of the best is passive.
    """
    rnd = np.random.RandomState(0)
    n = 50
    m = 4
    vals = -np.arange(1, n + 1)
    A = diags([vals], [0], (n, n))
    A = A.astype(np.float32)
    X = rnd.standard_normal((n, m))
    X = X.astype(np.float64)
    msg = 'Exited at iteration.*|Exited postprocessing with accuracies.*'
    for maxiter in range(1, 4):
        with pytest.warns(UserWarning, match=msg):
            _, _, l_h, r_h = lobpcg(A, X, tol=1e-08, maxiter=maxiter, retLambdaHistory=True, retResidualNormsHistory=True)
        assert_allclose(np.shape(l_h)[0], maxiter + 3)
        assert_allclose(np.shape(r_h)[0], maxiter + 3)
    with pytest.warns(UserWarning, match=msg):
        l, _, l_h, r_h = lobpcg(A, X, tol=1e-08, retLambdaHistory=True, retResidualNormsHistory=True)
    assert_allclose(np.shape(l_h)[0], 20 + 3)
    assert_allclose(np.shape(r_h)[0], 20 + 3)
    assert_allclose(l, l_h[-1])
    assert isinstance(l_h, list)
    assert isinstance(r_h, list)
    assert_allclose(np.shape(l_h), np.shape(np.asarray(l_h)))
    assert_allclose(np.shape(r_h), np.shape(np.asarray(r_h)))