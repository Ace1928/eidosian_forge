import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_allclose
import pandas as pd
import pytest
from statsmodels.tsa.statespace import dynamic_factor
from .results import results_varmax, results_dynamic_factor
from statsmodels.iolib.summary import forg
def test_start_params_nans():
    ix = pd.date_range('1960-01-01', '1982-10-01', freq='QS')
    dta = np.log(pd.DataFrame(results_varmax.lutkepohl_data, columns=['inv', 'inc', 'consump'], index=ix)).diff().iloc[1:]
    endog1 = dta.iloc[:-1]
    mod1 = dynamic_factor.DynamicFactor(endog1, k_factors=1, factor_order=1)
    endog2 = dta.copy()
    endog2.iloc[-1:] = np.nan
    mod2 = dynamic_factor.DynamicFactor(endog2, k_factors=1, factor_order=1)
    assert_allclose(mod2.start_params, mod1.start_params)