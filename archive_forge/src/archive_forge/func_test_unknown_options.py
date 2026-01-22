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
def test_unknown_options(self):
    c = np.array([-3, -2])
    A_ub = [[2, 1], [1, 1], [1, 0]]
    b_ub = [10, 8, 4]

    def f(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, options={}):
        linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=options)
    o = {key: self.options[key] for key in self.options}
    o['spam'] = 42
    assert_warns(OptimizeWarning, f, c, A_ub=A_ub, b_ub=b_ub, options=o)