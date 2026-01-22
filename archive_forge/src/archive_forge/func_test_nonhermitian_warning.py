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
@pytest.mark.filterwarnings('ignore:Exited at iteration 0')
@pytest.mark.filterwarnings('ignore:Exited postprocessing')
def test_nonhermitian_warning(capsys):
    """Check the warning of a Ritz matrix being not Hermitian
    by feeding a non-Hermitian input matrix.
    Also check stdout since verbosityLevel=1 and lack of stderr.
    """
    n = 10
    X = np.arange(n * 2).reshape(n, 2).astype(np.float32)
    A = np.arange(n * n).reshape(n, n).astype(np.float32)
    with pytest.warns(UserWarning, match='Matrix gramA'):
        _, _ = lobpcg(A, X, verbosityLevel=1, maxiter=0)
    out, err = capsys.readouterr()
    assert out.startswith('Solving standard eigenvalue')
    assert err == ''
    A += A.T
    _, _ = lobpcg(A, X, verbosityLevel=1, maxiter=0)
    out, err = capsys.readouterr()
    assert out.startswith('Solving standard eigenvalue')
    assert err == ''