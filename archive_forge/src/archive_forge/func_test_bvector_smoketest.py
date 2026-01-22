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
def test_bvector_smoketest(self):
    Adense = array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    As = csc_matrix(Adense)
    random.seed(1234)
    x = random.randn(3)
    b = As @ x
    x2 = spsolve(As, b)
    assert_array_almost_equal(x, x2)