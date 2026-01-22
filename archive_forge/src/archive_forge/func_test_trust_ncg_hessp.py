import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose
from scipy.optimize import (minimize, rosen, rosen_der, rosen_hess,
def test_trust_ncg_hessp(self):
    for x0 in (self.easy_guess, self.hard_guess, self.x_opt):
        r = minimize(rosen, x0, jac=rosen_der, hessp=rosen_hess_prod, tol=1e-08, method='trust-ncg')
        assert_allclose(self.x_opt, r['x'])