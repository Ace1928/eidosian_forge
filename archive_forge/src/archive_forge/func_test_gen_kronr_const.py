from typing import Tuple
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def test_gen_kronr_const(self):
    z_dims = (2, 2)
    for c_dims in TestKronLeftVar.C_DIMS:
        Z, C, L, prob = self.make_kron_prob(z_dims, c_dims, param=False, var_left=True, seed=0)
        prob.solve(solver='ECOS', abstol=1e-08, reltol=1e-08)
        self.assertEqual(prob.status, cp.OPTIMAL)
        con_viols = prob.constraints[0].violation()
        self.assertLessEqual(np.max(con_viols), 0.0001)
        self.assertItemsAlmostEqual(Z.value, L, places=4)