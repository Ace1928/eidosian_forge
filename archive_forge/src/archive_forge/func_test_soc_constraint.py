import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.reshape import reshape as reshape_atom
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_soc_constraint(self) -> None:
    exp = self.x + self.z
    scalar_exp = self.a + self.b
    constr = SOC(scalar_exp, exp)
    self.assertEqual(constr.size, 3)
    error_str = 'Argument dimensions (1,) and (1, 4), with axis=0, are incompatible.'
    with self.assertRaises(Exception) as cm:
        SOC(Variable(1), Variable((1, 4)))
    self.assertEqual(str(cm.exception), error_str)
    n = 5
    x0 = np.arange(n)
    t0 = 2
    x = cp.Variable(n, value=x0)
    t = cp.Variable(value=t0)
    resid = SOC(t, x).residual
    assert resid.ndim == 0
    dist = cp.sum_squares(x - x0) + cp.square(t - t0)
    prob = cp.Problem(cp.Minimize(dist), [SOC(t, x)])
    prob.solve()
    self.assertAlmostEqual(np.sqrt(dist.value), resid)
    n = 5
    k = 3
    x0 = np.arange(n * k).reshape((n, k))
    t0 = np.array([1, 2, 3])
    x = cp.Variable((n, k), value=x0)
    t = cp.Variable(k, value=t0)
    resid = SOC(t, x, axis=0).residual
    assert resid.shape == (k,)
    for i in range(k):
        dist = cp.sum_squares(x[:, i] - x0[:, i]) + cp.sum_squares(t[i] - t0[i])
        prob = cp.Problem(cp.Minimize(dist), [SOC(t[i], x[:, i])])
        prob.solve()
        self.assertAlmostEqual(np.sqrt(dist.value), resid[i])
    n = 5
    k = 3
    x0 = np.arange(n * k).reshape((k, n))
    t0 = np.array([1, 2, 3])
    x = cp.Variable((k, n), value=x0)
    t = cp.Variable(k, value=t0)
    resid = SOC(t, x, axis=1).residual
    assert resid.shape == (k,)
    for i in range(k):
        dist = cp.sum_squares(x[i, :] - x0[i, :]) + cp.sum_squares(t[i] - t0[i])
        prob = cp.Problem(cp.Minimize(dist), [SOC(t[i], x[i, :])])
        prob.solve()
        self.assertAlmostEqual(np.sqrt(dist.value), resid[i])
    k, n = (3, 3)
    x0 = np.ones((k, n))
    norms = np.linalg.norm(x0, ord=2)
    t0 = np.array([2, 0.5, -2]) * norms
    x = cp.Variable((k, n), value=x0)
    t = cp.Variable(k, value=t0)
    resid = SOC(t, x, axis=1).residual
    assert resid.shape == (k,)
    for i in range(k):
        dist = cp.sum_squares(x[i, :] - x0[i, :]) + cp.sum_squares(t[i] - t0[i])
        prob = cp.Problem(cp.Minimize(dist), [SOC(t[i], x[i, :])])
        prob.solve()
        self.assertAlmostEqual(np.sqrt(dist.value), resid[i], places=4)