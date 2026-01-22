import numpy as np
import pandas as pd
from numpy.testing import assert_, assert_equal, assert_allclose, assert_raises
from statsmodels.tsa.arima import specification, params
def test_set_params_single():
    exog = pd.DataFrame([[0]], columns=['a'])
    spec = specification.SARIMAXSpecification(exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    p = params.SARIMAXParams(spec=spec)

    def check(is_stationary='raise', is_invertible='raise'):
        assert_(not p.is_complete)
        assert_(not p.is_valid)
        if is_stationary == 'raise':
            assert_raises(ValueError, p.__getattribute__, 'is_stationary')
        else:
            assert_equal(p.is_stationary, is_stationary)
        if is_invertible == 'raise':
            assert_raises(ValueError, p.__getattribute__, 'is_invertible')
        else:
            assert_equal(p.is_invertible, is_invertible)
    p.exog_params = -6.0
    check()
    p.ar_params = -5.0
    check()
    p.ma_params = -4.0
    check()
    p.seasonal_ar_params = -3.0
    check(is_stationary=False)
    p.seasonal_ma_params = -2.0
    check(is_stationary=False, is_invertible=False)
    p.sigma2 = -1.0
    assert_(p.is_complete)
    assert_(not p.is_valid)
    assert_equal(p.params, [-6, -5, -4, -3, -2, -1])
    assert_equal(p.exog_params, [-6])
    assert_equal(p.ar_params, [-5])
    assert_equal(p.ma_params, [-4])
    assert_equal(p.seasonal_ar_params, [-3])
    assert_equal(p.seasonal_ma_params, [-2])
    assert_equal(p.sigma2, -1.0)
    assert_equal(p.ar_poly.coef, np.r_[1, 5])
    assert_equal(p.ma_poly.coef, np.r_[1, -4])
    assert_equal(p.seasonal_ar_poly.coef, np.r_[1, 0, 0, 0, 3])
    assert_equal(p.seasonal_ma_poly.coef, np.r_[1, 0, 0, 0, -2])
    assert_equal(p.reduced_ar_poly.coef, np.r_[1, 5, 0, 0, 3, 15])
    assert_equal(p.reduced_ma_poly.coef, np.r_[1, -4, 0, 0, -2, 8])
    p.exog_params = [1.0]
    p.ar_params = [2.0]
    p.ma_params = [3.0]
    p.seasonal_ar_params = [4.0]
    p.seasonal_ma_params = [5.0]
    p.sigma2 = [6.0]
    p.params = [1, 2, 3, 4, 5, 6]
    assert_equal(p.params, [1, 2, 3, 4, 5, 6])
    assert_equal(p.exog_params, [1])
    assert_equal(p.ar_params, [2])
    assert_equal(p.ma_params, [3])
    assert_equal(p.seasonal_ar_params, [4])
    assert_equal(p.seasonal_ma_params, [5])
    assert_equal(p.sigma2, 6.0)
    p.exog_params = np.array(6.0)
    p.ar_params = np.array(5.0)
    p.ma_params = np.array(4.0)
    p.seasonal_ar_params = np.array(3.0)
    p.seasonal_ma_params = np.array(2.0)
    p.sigma2 = np.array(1.0)
    assert_equal(p.params, [6, 5, 4, 3, 2, 1])
    assert_equal(p.exog_params, [6])
    assert_equal(p.ar_params, [5])
    assert_equal(p.ma_params, [4])
    assert_equal(p.seasonal_ar_params, [3])
    assert_equal(p.seasonal_ma_params, [2])
    assert_equal(p.sigma2, 1.0)
    p.params = [1, 2, 3, 4, 5, 6]
    assert_equal(p.params, [1, 2, 3, 4, 5, 6])
    assert_equal(p.exog_params, [1])
    assert_equal(p.ar_params, [2])
    assert_equal(p.ma_params, [3])
    assert_equal(p.seasonal_ar_params, [4])
    assert_equal(p.seasonal_ma_params, [5])
    assert_equal(p.sigma2, 6.0)
    assert_equal(p.ar_poly.coef, np.r_[1, -2])
    assert_equal(p.ma_poly.coef, np.r_[1, 3])
    assert_equal(p.seasonal_ar_poly.coef, np.r_[1, 0, 0, 0, -4])
    assert_equal(p.seasonal_ma_poly.coef, np.r_[1, 0, 0, 0, 5])
    assert_equal(p.reduced_ar_poly.coef, np.r_[1, -2, 0, 0, -4, 8])
    assert_equal(p.reduced_ma_poly.coef, np.r_[1, 3, 0, 0, 5, 15])