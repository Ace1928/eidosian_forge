import numpy as np
import pytest
import cvxpy as cvx
import cvxpy.problems.iterative as iterative
import cvxpy.settings as s
from cvxpy.lin_ops.tree_mat import prune_constants
from cvxpy.tests.base_test import BaseTest
def test_conv_prob(self) -> None:
    """Test a problem with convolution.
        """
    N = 5
    y = np.random.randn(N, 1)
    h = np.random.randn(2, 1)
    x = cvx.Variable((N, 1))
    v = cvx.conv(h, x)
    obj = cvx.Minimize(cvx.sum(cvx.multiply(y, v[0:N])))
    prob = cvx.Problem(obj, [])
    prob.solve(solver=cvx.ECOS)
    assert prob.status is cvx.UNBOUNDED
    y = np.random.randn(N)
    h = np.random.randn(2)
    x = cvx.Variable(N)
    v = cvx.convolve(h, x)
    obj = cvx.Minimize(cvx.sum(cvx.multiply(y, v[0:N])))
    prob = cvx.Problem(obj, [])
    prob.solve(solver=cvx.ECOS)
    assert prob.status is cvx.UNBOUNDED