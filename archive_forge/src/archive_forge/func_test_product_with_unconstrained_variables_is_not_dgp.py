import numpy as np
import scipy.sparse as sp
import cvxpy
from cvxpy.tests.base_test import BaseTest
def test_product_with_unconstrained_variables_is_not_dgp(self) -> None:
    x = cvxpy.Variable()
    y = cvxpy.Variable()
    prod = x * y
    self.assertTrue(not prod.is_dgp())
    self.assertTrue(not prod.is_log_log_convex())
    self.assertTrue(not prod.is_log_log_concave())
    z = cvxpy.Variable((), pos=True)
    prod = x * z
    self.assertTrue(not prod.is_dgp())
    self.assertTrue(not prod.is_log_log_convex())
    self.assertTrue(not prod.is_log_log_concave())