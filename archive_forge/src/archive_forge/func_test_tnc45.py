import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def test_tnc45(self):
    fg, x, bounds = (self.fg45, [2] * 5, [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
    xopt = [1, 2, 3, 4, 5]
    x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds, messages=optimize._tnc.MSG_NONE, maxfun=200)
    assert_allclose(self.f45(x), self.f45(xopt), atol=1e-08, err_msg='TNC failed with status: ' + optimize._tnc.RCSTRINGS[rc])