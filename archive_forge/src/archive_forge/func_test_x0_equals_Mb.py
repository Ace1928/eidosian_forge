import itertools
import platform
import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from numpy import zeros, arange, array, ones, eye, iscomplexobj
from numpy.linalg import norm
from scipy.sparse import spdiags, csr_matrix, kronsum
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._isolve import (bicg, bicgstab, cg, cgs,
def test_x0_equals_Mb(case):
    if case.solver is tfqmr:
        pytest.skip("Solver does not support x0='Mb'")
    A = case.A
    b = case.b
    x0 = 'Mb'
    rtol = 1e-08
    x, info = case.solver(A, b, x0=x0, rtol=rtol)
    assert_array_equal(x0, 'Mb')
    assert info == 0
    assert norm(A @ x - b) <= rtol * norm(b)