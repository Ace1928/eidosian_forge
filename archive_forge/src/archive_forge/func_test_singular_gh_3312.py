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
def test_singular_gh_3312(self):
    ij = np.array([(17, 0), (17, 6), (17, 12), (10, 13)], dtype=np.int32)
    v = np.array([0.284213, 0.94933781, 0.15767017, 0.38797296])
    A = csc_matrix((v, ij.T), shape=(20, 20))
    b = np.arange(20)
    try:
        with suppress_warnings() as sup:
            sup.filter(MatrixRankWarning, 'Matrix is exactly singular')
            x = spsolve(A, b)
        assert not np.isfinite(x).any()
    except RuntimeError:
        pass