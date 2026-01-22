import numpy as np
from numpy.testing import (
import pytest
import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.util import seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import VARProcess
from statsmodels.tsa.vector_ar.vecm import (
def test_VECM_seasonal_forecast():
    np.random.seed(964255)
    nobs = 200
    seasons = 6
    fact = np.cumsum(0.1 + np.random.randn(nobs, 2), 0)
    xx = np.random.randn(nobs + 2, 3)
    xx = xx[2:] + 0.6 * xx[1:-1] + 0.25 * xx[:-2]
    xx[:, :2] += fact[:, 0][:, None]
    xx[:, 2:] += fact[:, 1][:, None]
    xx += 3 * np.log(0.1 + np.arange(nobs)[:, None] % seasons)
    res0 = VECM(xx, k_ar_diff=0, coint_rank=2, deterministic='co', seasons=seasons, first_season=0).fit()
    res2 = VECM(xx, k_ar_diff=2, coint_rank=2, deterministic='co', seasons=seasons, first_season=0).fit()
    res4 = VECM(xx, k_ar_diff=4, coint_rank=2, deterministic='co', seasons=seasons, first_season=0).fit()
    assert_allclose(res2._delta_x.T[-2 * seasons:, -seasons:], res0._delta_x.T[-2 * seasons:, -seasons:], rtol=1e-10)
    assert_allclose(res4._delta_x.T[-2 * seasons:, -seasons:], res0._delta_x.T[-2 * seasons:, -seasons:], rtol=1e-10)
    assert_array_equal(np.argmin(res0.det_coef, axis=1), [1, 1, 1])
    assert_array_equal(np.argmin(res2.det_coef, axis=1), [1, 1, 1])
    assert_array_equal(np.argmin(res4.det_coef, axis=1), [1, 1, 1])
    dips_true = np.array([[4, 4, 4], [10, 10, 10], [16, 16, 16]])
    for res in [res0, res2, res4]:
        forecast = res.predict(steps=3 * seasons)
        dips = np.sort(np.argsort(forecast, axis=0)[:3], axis=0)
        assert_array_equal(dips, dips_true)