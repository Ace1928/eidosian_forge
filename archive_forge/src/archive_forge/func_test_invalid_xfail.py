import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises
from statsmodels.tsa.stattools import acovf
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import dowj, lake
from statsmodels.tsa.arima.estimators.yule_walker import yule_walker
@pytest.mark.xfail(reason='TODO: this does not raise an error due to the way linear_model.yule_walker works.')
def test_invalid_xfail():
    endog = np.arange(2) * 1.0
    assert_raises(ValueError, yule_walker, endog, ar_order=2)