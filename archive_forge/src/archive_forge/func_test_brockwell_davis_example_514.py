import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises
from statsmodels.tsa.stattools import acovf
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import dowj, lake
from statsmodels.tsa.arima.estimators.yule_walker import yule_walker
@pytest.mark.low_precision('Test against Example 5.1.4 in Brockwell and Davis (2016)')
def test_brockwell_davis_example_514():
    endog = lake.copy()
    res, _ = yule_walker(endog, ar_order=2, demean=True)
    assert_allclose(res.ar_params, [1.0538, -0.2668], atol=0.0001)
    assert_allclose(res.sigma2, 0.492, atol=0.0001)