import sys
import platform
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
from scipy._lib._util import VisibleDeprecationWarning
import scipy.sparse
import pytest
def test_solver_select(self):
    if has_cholmod:
        options = {'sparse': True, 'cholesky': True}
    elif has_umfpack:
        options = {'sparse': True, 'cholesky': False}
    else:
        options = {'sparse': True, 'cholesky': False, 'sym_pos': False}
    A, b, c = lpgen_2d(20, 20)
    res1 = linprog(c, A_ub=A, b_ub=b, method=self.method, options=options)
    res2 = linprog(c, A_ub=A, b_ub=b, method=self.method)
    assert_allclose(res1.fun, res2.fun, err_msg='linprog default solver unexpected result', rtol=2e-15, atol=1e-15)