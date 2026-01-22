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
@sup_sparse_efficiency
def test_superlu_dlamch_i386_nan(self):
    n = 8
    d = np.arange(n) + 1
    A = spdiags((d, 2 * d, d[::-1]), (-3, 0, 5), n, n)
    A = A.astype(np.float32)
    spilu(A)
    A = A + 1j * A
    B = A.A
    assert_(not np.isnan(B).any())