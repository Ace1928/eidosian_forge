import os
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from statsmodels.regression.linear_model import OLS, GLSAR
from statsmodels.tools.tools import add_constant
from statsmodels.datasets import macrodata
import statsmodels.stats.sandwich_covariance as sw
import statsmodels.stats.diagnostic as smsdia
import statsmodels.stats.outliers_influence as oi
def test_GLSARlag():
    from statsmodels.datasets import macrodata
    d2 = macrodata.load_pandas().data
    g_gdp = 400 * np.diff(np.log(d2['realgdp'].values))
    g_inv = 400 * np.diff(np.log(d2['realinv'].values))
    exogg = add_constant(np.c_[g_gdp, d2['realint'][:-1].values], prepend=False)
    mod1 = GLSAR(g_inv, exogg, 1)
    res1 = mod1.iterative_fit(5)
    mod4 = GLSAR(g_inv, exogg, 4)
    res4 = mod4.iterative_fit(10)
    assert_array_less(np.abs(res1.params / res4.params - 1), 0.03)
    assert_array_less(res4.ssr, res1.ssr)
    assert_array_less(np.abs(res4.bse / res1.bse) - 1, 0.015)
    assert_array_less(np.abs((res4.fittedvalues / res1.fittedvalues - 1).mean()), 0.015)
    assert_equal(len(mod4.rho), 4)