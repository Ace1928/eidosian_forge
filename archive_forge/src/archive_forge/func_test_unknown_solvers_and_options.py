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
def test_unknown_solvers_and_options():
    c = np.array([-3, -2])
    A_ub = [[2, 1], [1, 1], [1, 0]]
    b_ub = [10, 8, 4]
    assert_raises(ValueError, linprog, c, A_ub=A_ub, b_ub=b_ub, method='ekki-ekki-ekki')
    assert_raises(ValueError, linprog, c, A_ub=A_ub, b_ub=b_ub, method='highs-ekki')
    message = "Unrecognized options detected: {'rr_method': 'ekki-ekki-ekki'}"
    with pytest.warns(OptimizeWarning, match=message):
        linprog(c, A_ub=A_ub, b_ub=b_ub, options={'rr_method': 'ekki-ekki-ekki'})