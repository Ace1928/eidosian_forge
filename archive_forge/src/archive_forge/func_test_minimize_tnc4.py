import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def test_minimize_tnc4(self):
    x0, bnds = ([1.125, 0.125], [(1, None), (0, None)])
    xopt = [1, 0]
    x = optimize.minimize(self.f4, x0, method='TNC', jac=self.g4, bounds=bnds, options=self.opts).x
    assert_allclose(self.f4(x), self.f4(xopt), atol=1e-08)