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
def test_semi_continuous(self):
    c = np.array([1.0, 1.0, -1, -1])
    bounds = np.array([[0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [0.5, 1.5]])
    integrality = np.array([2, 3, 2, 3])
    res = linprog(c, bounds=bounds, integrality=integrality, method='highs')
    np.testing.assert_allclose(res.x, [0, 0, 1.5, 1])
    assert res.status == 0