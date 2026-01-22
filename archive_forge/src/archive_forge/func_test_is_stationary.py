import numpy as np
import pandas as pd
from numpy.testing import assert_, assert_equal, assert_allclose, assert_raises
from statsmodels.tsa.arima import specification, params
def test_is_stationary():
    spec = specification.SARIMAXSpecification(order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    p = params.SARIMAXParams(spec=spec)
    assert_raises(ValueError, p.__getattribute__, 'is_stationary')
    p.ar_params = [0.5]
    p.seasonal_ar_params = [0]
    assert_(p.is_stationary)
    p.ar_params = [1.0]
    assert_(not p.is_stationary)
    p.ar_params = [0]
    p.seasonal_ar_params = [0.5]
    assert_(p.is_stationary)
    p.seasonal_ar_params = [1.0]
    assert_(not p.is_stationary)
    p.ar_params = [0.2]
    p.seasonal_ar_params = [0.2]
    assert_(p.is_stationary)
    p.ar_params = [0.99]
    p.seasonal_ar_params = [0.99]
    assert_(p.is_stationary)
    p.ar_params = [1.0]
    p.seasonal_ar_params = [1.0]
    assert_(not p.is_stationary)