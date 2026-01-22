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
def test_enzo_example_c_with_degeneracy(self):
    m = 20
    c = -np.ones(m)
    tmp = 2 * np.pi * np.arange(1, m + 1) / (m + 1)
    A_eq = np.vstack((np.cos(tmp) - 1, np.sin(tmp)))
    b_eq = [0, 0]
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
    _assert_success(res, desired_fun=0, desired_x=np.zeros(m))