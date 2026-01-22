import numpy as np
import pytest
from numpy.testing import assert_allclose
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake
from statsmodels.tsa.arima.estimators.hannan_rissanen import (
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tools.tools import Bunch
def test_set_default_unbiased_with_fixed_params():
    endog = np.random.normal(size=1000)
    p_1, other_results_2 = hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=None, fixed_params={'ar.L1': 0.69607715})
    p_2, other_results_1 = hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=False, fixed_params={'ar.L1': 0.69607715})
    np.testing.assert_array_equal(p_1.ar_params, p_2.ar_params)
    np.testing.assert_array_equal(p_1.ma_params, p_2.ma_params)
    assert p_1.sigma2 == p_2.sigma2
    np.testing.assert_array_equal(other_results_1.resid, other_results_2.resid)