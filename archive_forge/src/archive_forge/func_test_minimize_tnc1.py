import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def test_minimize_tnc1(self):
    x0, bnds = ([-2, 1], ([-np.inf, None], [-1.5, None]))
    xopt = [1, 1]
    iterx = []
    res = optimize.minimize(self.f1, x0, method='TNC', jac=self.g1, bounds=bnds, options=self.opts, callback=iterx.append)
    assert_allclose(res.fun, self.f1(xopt), atol=1e-08)
    assert_equal(len(iterx), res.nit)