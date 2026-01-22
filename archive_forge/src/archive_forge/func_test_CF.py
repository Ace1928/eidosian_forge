import unittest
import numpy as np
from statsmodels.multivariate.factor_rotation._wrappers import rotate_factors
from statsmodels.multivariate.factor_rotation._gpa_rotation import (
from statsmodels.multivariate.factor_rotation._analytic_rotation import (
def test_CF(self):
    out = self.get_quartimax_example_derivative_free()
    A, table_required, L_required = out
    vgQ = lambda L=None, A=None, T=None: CF_objective(L=L, A=A, T=T, kappa=0, rotation_method='orthogonal', return_gradient=True)
    L, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
    self.assertTrue(np.allclose(L, L_required, atol=1e-05))
    ff = lambda L=None, A=None, T=None: CF_objective(L=L, A=A, T=T, kappa=0, rotation_method='orthogonal', return_gradient=False)
    L, phi, T, table = GPA(A, ff=ff, rotation_method='orthogonal')
    self.assertTrue(np.allclose(L, L_required, atol=1e-05))
    p, k = A.shape
    vgQ = lambda L=None, A=None, T=None: orthomax_objective(L=L, A=A, T=T, gamma=1, return_gradient=True)
    L_vm, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
    vgQ = lambda L=None, A=None, T=None: CF_objective(L=L, A=A, T=T, kappa=1 / p, rotation_method='orthogonal', return_gradient=True)
    L_CF, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
    ff = lambda L=None, A=None, T=None: CF_objective(L=L, A=A, T=T, kappa=1 / p, rotation_method='orthogonal', return_gradient=False)
    L_CF_df, phi, T, table = GPA(A, ff=ff, rotation_method='orthogonal')
    self.assertTrue(np.allclose(L_vm, L_CF, atol=1e-05))
    self.assertTrue(np.allclose(L_CF, L_CF_df, atol=1e-05))