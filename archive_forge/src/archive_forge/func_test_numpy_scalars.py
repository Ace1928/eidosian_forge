from __future__ import print_function
import unittest
import numpy as np
import cvxpy as cvx
import cvxpy.interface as intf
from cvxpy.reductions.solvers.conic_solvers import ecos_conif
from cvxpy.tests.base_test import BaseTest
def test_numpy_scalars(self) -> None:
    n = 6
    eps = 1e-06
    np.random.seed(10)
    P0 = np.random.randn(n, n)
    eye = np.eye(n)
    P0 = P0.T.dot(P0) + eps * eye
    print(P0)
    P1 = np.random.randn(n, n)
    P1 = P1.T.dot(P1)
    P2 = np.random.randn(n, n)
    P2 = P2.T.dot(P2)
    P3 = np.random.randn(n, n)
    P3 = P3.T.dot(P3)
    q0 = np.random.randn(n, 1)
    q1 = np.random.randn(n, 1)
    q2 = np.random.randn(n, 1)
    q3 = np.random.randn(n, 1)
    r0 = np.random.randn(1, 1)
    r1 = np.random.randn(1, 1)
    r2 = np.random.randn(1, 1)
    r3 = np.random.randn(1, 1)
    slack = cvx.Variable()
    x = cvx.Variable(n)
    objective = cvx.Minimize(0.5 * cvx.quad_form(x, P0) + q0.T @ x + r0 + slack)
    constraints = [0.5 * cvx.quad_form(x, P1) + q1.T @ x + r1 <= slack, 0.5 * cvx.quad_form(x, P2) + q2.T @ x + r2 <= slack, 0.5 * cvx.quad_form(x, P3) + q3.T @ x + r3 <= slack]
    p = cvx.Problem(objective, constraints)
    p.solve(solver=cvx.SCS)
    print(x.value)
    lam1 = constraints[0].dual_value
    lam2 = constraints[1].dual_value
    lam3 = constraints[2].dual_value
    print(type(lam1))
    P_lam = P0 + lam1 * P1 + lam2 * P2 + lam3 * P3
    q_lam = q0 + lam1 * q1 + lam2 * q2 + lam3 * q3
    r_lam = r0 + lam1 * r1 + lam2 * r2 + lam3 * r3
    dual_result = -0.5 * q_lam.T.dot(P_lam).dot(q_lam) + r_lam
    print(dual_result.shape)
    self.assertEqual(intf.shape(dual_result), (1, 1))