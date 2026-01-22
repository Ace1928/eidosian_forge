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
def test_dtype_cast(self):
    A_real = scipy.sparse.csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
    A_complex = scipy.sparse.csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5 + 1j]])
    b_real = np.array([1, 1, 1])
    b_complex = np.array([1, 1, 1]) + 1j * np.array([1, 1, 1])
    x = spsolve(A_real, b_real)
    assert_(np.issubdtype(x.dtype, np.floating))
    x = spsolve(A_real, b_complex)
    assert_(np.issubdtype(x.dtype, np.complexfloating))
    x = spsolve(A_complex, b_real)
    assert_(np.issubdtype(x.dtype, np.complexfloating))
    x = spsolve(A_complex, b_complex)
    assert_(np.issubdtype(x.dtype, np.complexfloating))