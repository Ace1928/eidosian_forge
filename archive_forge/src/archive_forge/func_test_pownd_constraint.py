import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.reshape import reshape as reshape_atom
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_pownd_constraint(self) -> None:
    n = 4
    W, z = (Variable(n), Variable())
    np.random.seed(0)
    alpha = 0.5 + np.random.rand(n)
    alpha /= np.sum(alpha)
    with self.assertRaises(ValueError):
        con = PowConeND(W, z, alpha + 0.01)
    with self.assertRaises(ValueError):
        con = PowConeND(W, z, alpha.reshape((n, 1)))
    with self.assertRaises(ValueError):
        con = PowConeND(reshape_atom(W, (n, 1)), z, alpha.reshape((n, 1)), axis=1)
    con = PowConeND(W, z, alpha)
    W0 = 0.1 + np.random.rand(n)
    z0 = np.prod(np.power(W0, alpha)) + 0.05
    W.value, z.value = (W0, z0)
    viol = con.violation()
    self.assertGreaterEqual(viol, 0.01)
    self.assertLessEqual(viol, 0.06)