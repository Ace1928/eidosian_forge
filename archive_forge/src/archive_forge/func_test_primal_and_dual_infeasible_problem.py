import osqp
from osqp._osqp import constant
import numpy as np
from scipy import sparse
import unittest
def test_primal_and_dual_infeasible_problem(self):
    self.n = 2
    self.m = 4
    self.P = sparse.csc_matrix((2, 2))
    self.q = np.array([-1.0, -1.0])
    self.A = sparse.csc_matrix([[1.0, -1.0], [-1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    self.l = np.array([1.0, 1.0, 0.0, 0.0])
    self.u = np.inf * np.ones(self.m)
    self.model = osqp.OSQP()
    self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **self.opts)
    x0 = 25.0 * np.ones(self.n)
    y0 = -2.0 * np.ones(self.m)
    self.model.warm_start(x=x0, y=y0)
    res = self.model.solve()
    self.assertIn(res.info.status_val, [constant('OSQP_PRIMAL_INFEASIBLE'), constant('OSQP_DUAL_INFEASIBLE')])