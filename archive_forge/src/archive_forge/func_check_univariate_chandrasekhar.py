import numpy as np
import pandas as pd
from .results import results_varmax
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.kalman_filter import (
from numpy.testing import assert_allclose
import pytest
def check_univariate_chandrasekhar(filter_univariate=False, **kwargs):
    index = pd.date_range('1960-01-01', '1982-10-01', freq='QS')
    dta = pd.DataFrame(results_varmax.lutkepohl_data, columns=['inv', 'inc', 'consump'], index=index)
    endog = np.log(dta['inv']).diff().loc['1960-04-01':'1978-10-01']
    mod_orig = sarimax.SARIMAX(endog, **kwargs)
    mod_chand = sarimax.SARIMAX(endog, **kwargs)
    mod_chand.ssm.filter_chandrasekhar = True
    params = mod_orig.start_params
    mod_orig.ssm.filter_univariate = filter_univariate
    mod_chand.ssm.filter_univariate = filter_univariate
    res_chand = mod_chand.smooth(params)
    res_orig = mod_orig.smooth(params)
    check_output(res_chand, res_orig)