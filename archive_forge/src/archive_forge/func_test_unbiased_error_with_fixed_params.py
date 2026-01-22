import numpy as np
import pytest
from numpy.testing import assert_allclose
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake
from statsmodels.tsa.arima.estimators.hannan_rissanen import (
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tools.tools import Bunch
def test_unbiased_error_with_fixed_params():
    endog = np.random.normal(size=1000)
    msg = 'Third step of Hannan-Rissanen estimation to remove parameter bias is not yet implemented for the case with fixed parameters.'
    with pytest.raises(NotImplementedError, match=msg):
        hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=True, fixed_params={'ar.L1': 0})