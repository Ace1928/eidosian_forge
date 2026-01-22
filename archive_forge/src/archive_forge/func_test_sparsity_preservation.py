import sys
import threading
import numpy as np
from numpy import array, finfo, arange, eye, all, unique, ones, dot
import numpy.random as random
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
import scipy.linalg
from scipy.linalg import norm, inv
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
from scipy.sparse.linalg import SuperLU
from scipy.sparse.linalg._dsolve import (spsolve, use_solver, splu, spilu,
import scipy.sparse
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import ComplexWarning
def test_sparsity_preservation(self):
    ident = csc_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    b = csc_matrix([[0, 1], [1, 0], [0, 0]])
    x = spsolve(ident, b)
    assert_equal(ident.nnz, 3)
    assert_equal(b.nnz, 2)
    assert_equal(x.nnz, 2)
    assert_allclose(x.A, b.A, atol=1e-12, rtol=1e-12)