import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from statsmodels.tsa.statespace import varmax
from .results import results_var_R
def test_var_ct_as_exog0():
    test = 'ct_as_exog0'
    results = results_var_R.res_ct_as_exog0
    mod = varmax.VARMAX(endog, order=(2, 0), exog=exog0[:, :2], trend='n', loglikelihood_burn=2)
    res = mod.smooth(results['params'])
    assert_allclose(res.llf, results['llf'])
    columns = ['{}.fcast.{}.fcst'.format(test, name) for name in endog.columns]
    assert_allclose(res.forecast(10, exog=exog0_fcast[:, :2]), results_var_R_output[columns].iloc[:10])
    check_irf(test, mod, results)