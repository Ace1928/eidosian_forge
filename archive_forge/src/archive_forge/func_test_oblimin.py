import unittest
import numpy as np
from statsmodels.multivariate.factor_rotation._wrappers import rotate_factors
from statsmodels.multivariate.factor_rotation._gpa_rotation import (
from statsmodels.multivariate.factor_rotation._analytic_rotation import (
def test_oblimin(self):
    A, table_required, L_required = self.get_quartimin_example()
    vgQ = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=0, rotation_method='oblique')
    L, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='oblique')
    self.assertTrue(np.allclose(table, table_required, atol=1e-05))
    self.assertTrue(np.allclose(L, L_required, atol=1e-05))
    ff = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=0, rotation_method='oblique', return_gradient=False)
    L, phi, T, table = GPA(A, ff=ff, rotation_method='oblique')
    self.assertTrue(np.allclose(L, L_required, atol=1e-05))
    self.assertTrue(np.allclose(table, table_required, atol=1e-05))
    A, table_required, L_required = self.get_biquartimin_example()
    vgQ = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=1 / 2, rotation_method='oblique')
    L, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='oblique')
    self.assertTrue(np.allclose(table, table_required, atol=1e-05))
    self.assertTrue(np.allclose(L, L_required, atol=1e-05))
    out = self.get_biquartimin_example_derivative_free()
    A, table_required, L_required = out
    ff = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=1 / 2, rotation_method='oblique', return_gradient=False)
    L, phi, T, table = GPA(A, ff=ff, rotation_method='oblique')
    self.assertTrue(np.allclose(L, L_required, atol=1e-05))
    self.assertTrue(np.allclose(table, table_required, atol=1e-05))