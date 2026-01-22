from statsmodels.compat.pandas import QUARTER_END
import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.io import matlab
from statsmodels.tsa.statespace import dynamic_factor_mq, initialization
@pytest.mark.parametrize('run', ['news_112', 'news_222', 'news_block_112', 'news_block_222'])
def test_news(matlab_results, run):
    endog_M, endog_Q = matlab_results[:2]
    results = matlab_results[2][run]
    updated_M, updated_Q = matlab_results[-2:]
    mod1 = dynamic_factor_mq.DynamicFactorMQ(endog_M.iloc[:, :results['k_endog_M']], endog_quarterly=endog_Q, factors=results['factors'], factor_orders=results['factor_orders'], factor_multiplicities=results['factor_multiplicities'], idiosyncratic_ar1=True, init_t0=True, obs_cov_diag=True, standardize=True)
    mod1.initialize_known(results['initial_state'], results['initial_state_cov'])
    res1 = mod1.smooth(results['params'], cov_type='none')
    res2 = res1.apply(updated_M.iloc[:, :results['k_endog_M']], endog_quarterly=updated_Q, retain_standardization=True)
    news = res2.news(res1, impact_date='2016-09', comparison_type='previous')
    assert_allclose(news.revision_impacts.loc['2016-09', 'GDPC1'], results['revision_impacts'])
    columns = ['forecast (prev)', 'observed', 'weight', 'impact']
    actual = news.details_by_impact.loc['2016-09', 'GDPC1'][columns]
    assert_allclose(actual.loc['2016-06', 'CPIAUCSL'], results['news_table'][0])
    assert_allclose(actual.loc['2016-06', 'UNRATE'], results['news_table'][1])
    assert_allclose(actual.loc['2016-06', 'PAYEMS'], results['news_table'][2])
    if mod1.k_endog_M == 6:
        i = 6
        assert_allclose(actual.loc['2016-06', 'RSAFS'], results['news_table'][3])
        assert_allclose(actual.loc['2016-05', 'TTLCONS'], results['news_table'][4])
        assert_allclose(actual.loc['2016-06', 'TCU'], results['news_table'][5])
    else:
        i = 3
    assert_allclose(actual.loc['2016-06', 'GDPC1'], results['news_table'][i])