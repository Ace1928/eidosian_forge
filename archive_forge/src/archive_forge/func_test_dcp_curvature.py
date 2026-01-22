import cvxpy as cvx
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_dcp_curvature(self) -> None:
    expr = 1 + cvx.exp(cvx.Variable())
    self.assertEqual(expr.curvature, s.CONVEX)
    expr = cvx.Parameter() * cvx.Variable(nonneg=True)
    self.assertEqual(expr.curvature, s.AFFINE)
    f = lambda x: x ** 2 + x ** 0.5
    expr = f(cvx.Constant(2))
    self.assertEqual(expr.curvature, s.CONSTANT)
    expr = cvx.exp(cvx.Variable()) ** 2
    self.assertEqual(expr.curvature, s.CONVEX)
    expr = 1 - cvx.sqrt(cvx.Variable())
    self.assertEqual(expr.curvature, s.CONVEX)
    expr = cvx.log(cvx.sqrt(cvx.Variable()))
    self.assertEqual(expr.curvature, s.CONCAVE)
    expr = -cvx.exp(cvx.Variable()) ** 2
    self.assertEqual(expr.curvature, s.CONCAVE)
    expr = cvx.log(cvx.exp(cvx.Variable()))
    self.assertEqual(expr.is_dcp(), False)
    expr = cvx.entr(cvx.Variable(nonneg=True))
    self.assertEqual(expr.curvature, s.CONCAVE)
    expr = ((cvx.Variable() ** 2) ** 0.5) ** 0
    self.assertEqual(expr.curvature, s.CONSTANT)