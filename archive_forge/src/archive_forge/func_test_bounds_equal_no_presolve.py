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
def test_bounds_equal_no_presolve(self):
    c = [1, 2]
    A_ub = [[1, 2], [1.1, 2.2]]
    b_ub = [4, 8]
    bounds = [(1, 2), (2, 2)]
    o = {key: self.options[key] for key in self.options}
    o['presolve'] = False
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=o)
    _assert_infeasible(res)