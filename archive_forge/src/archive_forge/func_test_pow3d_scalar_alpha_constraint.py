import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.reshape import reshape as reshape_atom
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_pow3d_scalar_alpha_constraint(self) -> None:
    """
        Simple test case with scalar AND vector `alpha`
        inputs to `PowCone3D`
        """
    x_0 = cp.Variable(shape=(3,))
    x = cp.Variable(shape=(3,))
    cons = [cp.PowCone3D(x_0[0], x_0[1], x_0[2], 0.25), x <= -10]
    obj = cp.Minimize(cp.norm(x - x_0))
    prob = cp.Problem(obj, cons)
    prob.solve()
    self.assertAlmostEqual(prob.value, 17.320508075380552)