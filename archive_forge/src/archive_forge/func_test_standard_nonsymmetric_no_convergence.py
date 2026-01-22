import threading
import itertools
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from pytest import raises as assert_raises
import pytest
from numpy import dot, conj, random
from scipy.linalg import eig, eigh
from scipy.sparse import csc_matrix, csr_matrix, diags, rand
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._eigen.arpack import (eigs, eigsh, arpack,
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
def test_standard_nonsymmetric_no_convergence():
    np.random.seed(1234)
    m = generate_matrix(30, complex_=True)
    tol, rtol, atol = _get_test_tolerance('d')
    try:
        w, v = eigs(m, 4, which='LM', v0=m[:, 0], maxiter=5, tol=tol)
        raise AssertionError('Spurious no-error exit')
    except ArpackNoConvergence as err:
        k = len(err.eigenvalues)
        if k <= 0:
            raise AssertionError('Spurious no-eigenvalues-found case') from err
        w, v = (err.eigenvalues, err.eigenvectors)
        for ww, vv in zip(w, v.T):
            assert_allclose(dot(m, vv), ww * vv, rtol=rtol, atol=atol)