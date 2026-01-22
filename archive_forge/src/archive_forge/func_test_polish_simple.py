import osqp
from osqp.tests.utils import load_high_accuracy, rel_tol, abs_tol, decimal_tol
import numpy as np
from scipy import sparse
import unittest
import numpy.testing as nptest
def test_polish_simple(self):
    self.P = sparse.diags([11.0, 0.0], format='csc')
    self.q = np.array([3, 4])
    self.A = sparse.csc_matrix([[-1, 0], [0, -1], [-1, -3], [2, 5], [3, 4]])
    self.u = np.array([0, 0, -15, 100, 80])
    self.l = -100000.0 * np.ones(len(self.u))
    self.n = self.P.shape[0]
    self.m = self.A.shape[0]
    self.model = osqp.OSQP()
    self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **self.opts)
    res = self.model.solve()
    x_sol, y_sol, obj_sol = load_high_accuracy('test_polish_simple')
    nptest.assert_allclose(res.x, x_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_allclose(res.y, y_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_almost_equal(res.info.obj_val, obj_sol, decimal=decimal_tol)