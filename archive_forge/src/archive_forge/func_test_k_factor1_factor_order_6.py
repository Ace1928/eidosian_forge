from statsmodels.compat.pandas import QUARTER_END
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tsa.statespace import (
@pytest.mark.filterwarnings('ignore:Log-likelihood decreased')
def test_k_factor1_factor_order_6(reset_randomstate):
    endog_M, endog_Q, _ = gen_k_factor1(nobs=100, idiosyncratic_var=0.0)
    mod = dynamic_factor_mq.DynamicFactorMQ(endog_M, endog_quarterly=endog_Q, factor_orders=6, idiosyncratic_ar1=False, standardize=False)
    mod.fit()