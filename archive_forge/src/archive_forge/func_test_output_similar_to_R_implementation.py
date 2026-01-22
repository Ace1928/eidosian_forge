from pathlib import Path
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import MSTL
def test_output_similar_to_R_implementation(data_pd, mstl_results):
    mod = MSTL(endog=data_pd, periods=(24, 24 * 7), stl_kwargs={'seasonal_deg': 0, 'seasonal_jump': 1, 'trend_jump': 1, 'trend_deg': 1, 'low_pass_jump': 1, 'low_pass_deg': 1, 'inner_iter': 2, 'outer_iter': 0})
    res = mod.fit()
    expected_observed = mstl_results['Data']
    expected_trend = mstl_results['Trend']
    expected_seasonal = mstl_results[['Seasonal24', 'Seasonal168']]
    expected_resid = mstl_results['Remainder']
    assert_allclose(res.observed, expected_observed)
    assert_allclose(res.trend, expected_trend)
    assert_allclose(res.seasonal, expected_seasonal)
    assert_allclose(res.resid, expected_resid)