import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.reshape import reshape as reshape_atom
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_nonneg_dual(self) -> None:
    x = cp.Variable(3)
    c = np.arange(3)
    objective = cp.Minimize(cp.sum(x))
    prob = cp.Problem(objective, [c - x <= 0])
    prob.solve(solver=cp.ECOS)
    dual = prob.constraints[0].dual_value
    prob = cp.Problem(objective, [cp.NonNeg(x - c)])
    prob.solve(solver=cp.ECOS)
    self.assertItemsAlmostEqual(prob.constraints[0].dual_value, dual)
    prob.solve(solver=cp.OSQP)
    self.assertItemsAlmostEqual(prob.constraints[0].dual_value, dual)