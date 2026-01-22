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
def test_mip3(self):
    A_ub = np.array([[-1, 1], [3, 2], [2, 3]])
    b_ub = np.array([1, 12, 12])
    c = -np.array([0, 1])
    bounds = [(0, np.inf)] * len(c)
    integrality = [1] * len(c)
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=self.method, integrality=integrality)
    np.testing.assert_allclose(res.fun, -2)
    assert np.allclose(res.x, [1, 2]) or np.allclose(res.x, [2, 2])