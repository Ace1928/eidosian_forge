from __future__ import print_function
import unittest
import numpy as np
import cvxpy as cvx
import cvxpy.interface as intf
from cvxpy.reductions.solvers.conic_solvers import ecos_conif
from cvxpy.tests.base_test import BaseTest
def test_portfolio_problem(self) -> None:
    """Test portfolio problem that caused dcp_attr errors.
        """
    import numpy as np
    import scipy.sparse as sp
    np.random.seed(5)
    n = 100
    m = 10
    F = sp.rand(m, n, density=0.01)
    F.data = np.ones(len(F.data))
    D = sp.eye(n).tocoo()
    D.data = np.random.randn(len(D.data)) ** 2
    Z = np.random.randn(m, 1)
    Z = Z.dot(Z.T)
    x = cvx.Variable(n)
    y = F @ x
    cvx.square(cvx.norm(D @ x)) + cvx.square(Z @ y)