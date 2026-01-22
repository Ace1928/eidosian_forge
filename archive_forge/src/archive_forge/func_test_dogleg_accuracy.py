import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose
from scipy.optimize import (minimize, rosen, rosen_der, rosen_hess,
def test_dogleg_accuracy(self):
    x0 = self.hard_guess
    r = minimize(rosen, x0, jac=rosen_der, hess=rosen_hess, tol=1e-08, method='dogleg', options={'return_all': True})
    assert_allclose(x0, r['allvecs'][0])
    assert_allclose(r['x'], r['allvecs'][-1])
    assert_allclose(r['x'], self.x_opt)