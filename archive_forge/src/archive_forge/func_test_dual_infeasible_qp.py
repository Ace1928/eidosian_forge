import osqp
from osqp._osqp import constant
import numpy as np
from scipy import sparse
import unittest
def test_dual_infeasible_qp(self):
    self.P = sparse.diags([4.0, 0.0], format='csc')
    self.q = np.array([0, 2])
    self.A = sparse.csc_matrix([[1.0, 1.0], [-1.0, 1.0]])
    self.l = np.array([-np.inf, -np.inf])
    self.u = np.array([2.0, 3.0])
    self.model = osqp.OSQP()
    self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **self.opts)
    res = self.model.solve()
    self.assertEqual(res.info.status_val, constant('OSQP_DUAL_INFEASIBLE'))