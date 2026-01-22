import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import statsmodels.api as sm
from statsmodels.tools import numdiff
from statsmodels.tools.numdiff import (
def test_grad_fun1_cs(self):
    for test_params in self.params:
        gtrue = self.gradtrue(test_params)
        fun = self.fun()
        gcs = numdiff.approx_fprime_cs(test_params, fun, args=self.args)
        assert_almost_equal(gtrue, gcs, decimal=DEC13)