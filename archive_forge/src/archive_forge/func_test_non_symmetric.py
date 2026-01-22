from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_equal
import cvxpy as cp
from cvxpy.settings import EIGVAL_TOL
from cvxpy.tests.base_test import BaseTest
def test_non_symmetric(self) -> None:
    """Test when P is constant and not symmetric.
        """
    P = np.array([[2, 2], [3, 4]])
    x = cp.Variable(2)
    with self.assertRaises(Exception) as cm:
        cp.quad_form(x, P)
    self.assertTrue('Quadratic form matrices must be symmetric/Hermitian.' in str(cm.exception))