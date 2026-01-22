import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from statsmodels.sandbox.nonparametric import smoothers
from statsmodels.regression.linear_model import OLS, WLS
def test_coef(self):
    assert_almost_equal(self.res_ps.coef.ravel(), self.res2.params, decimal=14)