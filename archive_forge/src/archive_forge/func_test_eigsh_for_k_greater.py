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
def test_eigsh_for_k_greater():
    A_sparse = diags([1, -2, 1], [-1, 0, 1], shape=(4, 4))
    A = generate_matrix(4, sparse=False)
    M_dense = generate_matrix_symmetric(4, pos_definite=True)
    M_sparse = generate_matrix_symmetric(4, pos_definite=True, sparse=True)
    M_linop = aslinearoperator(M_dense)
    eig_tuple1 = eigh(A, b=M_dense)
    eig_tuple2 = eigh(A, b=M_sparse)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        assert_equal(eigsh(A, M=M_dense, k=4), eig_tuple1)
        assert_equal(eigsh(A, M=M_dense, k=5), eig_tuple1)
        assert_equal(eigsh(A, M=M_sparse, k=5), eig_tuple2)
        assert_raises(TypeError, eigsh, A, M=M_linop, k=4)
        assert_raises(TypeError, eigsh, aslinearoperator(A), k=4)
        assert_raises(TypeError, eigsh, A_sparse, M=M_dense, k=4)