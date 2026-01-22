import numpy as np
import pandas as pd
from numpy.testing import assert_, assert_equal, assert_allclose, assert_raises
from statsmodels.tsa.arima import specification, params
def test_set_poly_short_lags_nonconsecutive():
    exog = pd.DataFrame([[0, 0]], columns=['a', 'b'])
    spec = specification.SARIMAXSpecification(exog=exog, order=([0, 1], 1, [0, 1]), seasonal_order=([0, 1], 1, [0, 1], 4))
    p = params.SARIMAXParams(spec=spec)
    p.ar_poly = [1, 0, -0.5]
    assert_equal(p.ar_params, [0.5])
    p.ar_poly = np.polynomial.Polynomial([1, 0, -0.55])
    assert_equal(p.ar_params, [0.55])
    p.ma_poly = [1, 0, 0.3]
    assert_equal(p.ma_params, [0.3])
    p.ma_poly = np.polynomial.Polynomial([1, 0, 0.35])
    assert_equal(p.ma_params, [0.35])
    p.seasonal_ar_poly = [1, 0, 0, 0, 0, 0, 0, 0, -0.2]
    assert_equal(p.seasonal_ar_params, [0.2])
    p.seasonal_ar_poly = np.polynomial.Polynomial([1, 0, 0, 0, 0, 0, 0, 0, -0.25])
    assert_equal(p.seasonal_ar_params, [0.25])
    p.seasonal_ma_poly = [1, 0, 0, 0, 0, 0, 0, 0, 0.1]
    assert_equal(p.seasonal_ma_params, [0.1])
    p.seasonal_ma_poly = np.polynomial.Polynomial([1, 0, 0, 0, 0, 0, 0, 0, 0.15])
    assert_equal(p.seasonal_ma_params, [0.15])
    assert_raises(ValueError, p.__setattr__, 'ar_poly', [1, 1, -0.5])
    assert_raises(ValueError, p.__setattr__, 'ma_poly', [1, 1, 0.3])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ar_poly', [1, 0, 0, 0, 1.0, 0, 0, 0, -0.2])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ma_poly', [1, 0, 0, 0, 1.0, 0, 0, 0, 0.1])