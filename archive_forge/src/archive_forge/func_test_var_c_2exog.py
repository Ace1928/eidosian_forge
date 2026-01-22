import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from statsmodels.tsa.statespace import varmax
from .results import results_var_R
def test_var_c_2exog():
    test = 'c_2exog'
    results = results_var_R.res_c_2exog
    exog = dta[['inc', 'inv']].loc['1960Q2':'1978']
    exog_fcast = dta[['inc', 'inv']].loc['1979Q1':'1981Q2']
    mod = varmax.VARMAX(endog, order=(2, 0), exog=exog, trend='c', loglikelihood_burn=2)
    res = mod.smooth(results['params'])
    assert_allclose(res.llf, results['llf'])
    columns = ['{}.fcast.{}.fcst'.format(test, name) for name in endog.columns]
    assert_allclose(res.forecast(10, exog=exog_fcast), results_var_R_output[columns].iloc[:10])
    check_irf(test, mod, results)