import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def test_minimize_tnc1c(self):
    x0, bnds = ([-2, 1], ([-np.inf, None], [-1.5, None]))
    xopt = [1, 1]
    x = optimize.minimize(self.fg1, x0, method='TNC', jac=True, bounds=bnds, options=self.opts).x
    assert_allclose(self.f1(x), self.f1(xopt), atol=1e-08)