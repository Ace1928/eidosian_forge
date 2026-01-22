import numpy as np
import pytest
from numpy.testing import assert_allclose
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake
from statsmodels.tsa.arima.estimators.hannan_rissanen import (
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tools.tools import Bunch
@pytest.mark.xfail(reason='TODO: improve checks on valid order parameters.')
def test_initial_order():
    endog = np.arange(20) * 1.0
    hannan_rissanen(endog, ar_order=2, ma_order=0, initial_ar_order=1)
    hannan_rissanen(endog, ar_order=0, ma_order=2, initial_ar_order=1)
    hannan_rissanen(endog, ar_order=0, ma_order=2, initial_ar_order=20)