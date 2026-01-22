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
def test_mip4(self):
    A_ub = np.array([[-1, -2], [-4, -1], [2, 1]])
    b_ub = np.array([14, -33, 20])
    c = np.array([8, 1])
    bounds = [(0, np.inf)] * len(c)
    integrality = [0, 1]
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=self.method, integrality=integrality)
    np.testing.assert_allclose(res.x, [6.5, 7])
    np.testing.assert_allclose(res.fun, 59)