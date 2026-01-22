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
def test_mip5(self):
    A_ub = np.array([[1, 1, 1]])
    b_ub = np.array([7])
    A_eq = np.array([[4, 2, 1]])
    b_eq = np.array([12])
    c = np.array([-3, -2, -1])
    bounds = [(0, np.inf), (0, np.inf), (0, 1)]
    integrality = [0, 1, 0]
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, integrality=integrality)
    np.testing.assert_allclose(res.x, [0, 6, 0])
    np.testing.assert_allclose(res.fun, -12)
    assert res.get('mip_node_count', None) is not None
    assert res.get('mip_dual_bound', None) is not None
    assert res.get('mip_gap', None) is not None