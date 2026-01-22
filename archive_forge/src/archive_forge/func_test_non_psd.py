from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_equal
import cvxpy as cp
from cvxpy.settings import EIGVAL_TOL
from cvxpy.tests.base_test import BaseTest
def test_non_psd(self) -> None:
    """Test error when P is symmetric but not definite.
        """
    P = np.array([[1, 0], [0, -1]])
    x = cp.Variable(2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cost = cp.quad_form(x, P)
    prob = cp.Problem(cp.Minimize(cost), [x == [1, 2]])
    with self.assertRaises(Exception) as cm:
        prob.solve(solver=cp.SCS)
    self.assertTrue('Problem does not follow DCP rules.' in str(cm.exception))