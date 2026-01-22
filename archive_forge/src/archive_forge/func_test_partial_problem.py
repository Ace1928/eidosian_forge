import numpy as np
import cvxpy as cp
from cvxpy import Maximize, Minimize, Problem
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms.partial_optimize import partial_optimize
def test_partial_problem(self) -> None:
    """Test domain for partial minimization/maximization problems.
        """
    for obj in [Minimize(self.a ** (-1)), Maximize(cp.log(self.a))]:
        orig_prob = Problem(obj, [self.x + self.a >= [5, 8]])
        expr = partial_optimize(orig_prob, dont_opt_vars=[self.x, self.a])
        dom = expr.domain
        constr = [self.a >= -100, self.x >= 0]
        prob = Problem(Minimize(sum(self.x + self.a)), dom + constr)
        prob.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(prob.value, 13)
        assert self.a.value >= 0
        assert np.all((self.x + self.a - [5, 8]).value >= -0.001)
        expr = partial_optimize(orig_prob, opt_vars=[self.x])
        dom = expr.domain
        constr = [self.a >= -100, self.x >= 0]
        prob = Problem(Minimize(sum(self.x + self.a)), dom + constr)
        prob.solve(solver=cp.ECOS)
        self.assertAlmostEqual(prob.value, 0)
        assert self.a.value >= -0.001
        self.assertItemsAlmostEqual(self.x.value, [0, 0])
        expr = partial_optimize(orig_prob, opt_vars=[self.x, self.a])
        dom = expr.domain
        constr = [self.a >= -100, self.x >= 0]
        prob = Problem(Minimize(sum(self.x + self.a)), dom + constr)
        prob.solve(solver=cp.ECOS)
        self.assertAlmostEqual(self.a.value, -100)
        self.assertItemsAlmostEqual(self.x.value, [0, 0])