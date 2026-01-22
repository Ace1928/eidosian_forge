import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def check_dual_domains(self, places) -> None:
    for con in self.constraints:
        if isinstance(con, cp.constraints.Cone):
            dual_violation = con.dual_residual
            if isinstance(con, cp.constraints.SOC):
                dual_violation = np.linalg.norm(dual_violation)
            self.tester.assertLessEqual(dual_violation, 10 ** (-places))
        elif isinstance(con, cp.constraints.Inequality):
            dv = con.dual_value
            min_dv = np.min(dv)
            self.tester.assertGreaterEqual(min_dv, -10 ** (-places))
        elif isinstance(con, (cp.constraints.Equality, cp.constraints.Zero)):
            dv = con.dual_value
            self.tester.assertIsNotNone(dv)
            if isinstance(dv, np.ndarray):
                contents = dv.dtype
                self.tester.assertEqual(contents, float)
            else:
                self.tester.assertIsInstance(dv, float)
        else:
            raise ValueError('Unknown constraint type %s.' % type(con))