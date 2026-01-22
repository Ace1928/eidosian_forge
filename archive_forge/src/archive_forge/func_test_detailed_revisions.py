from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
@pytest.mark.parametrize('revisions_details_start', [True, -10, 200])
def test_detailed_revisions(revisions_details_start):
    y = np.log(dta[['realgdp', 'realcons', 'realinv', 'cpi']]).diff().iloc[1:] * 100
    y.iloc[-1, 0] = np.nan
    y_revised = y.copy()
    revisions = {('2009Q2', 'realgdp'): 1.1, ('2009Q3', 'realcons'): 0.5, ('2009Q2', 'realinv'): -0.3, ('2009Q2', 'cpi'): 0.2, ('2009Q3', 'cpi'): 0.2}
    for key, diff in revisions.items():
        y_revised.loc[key] += diff
    mod = varmax.VARMAX(y, trend='n')
    ar_coeff = {'realgdp': 0.9, 'realcons': 0.8, 'realinv': 0.7, 'cpi': 0.6}
    params = np.r_[np.diag(list(ar_coeff.values())).flatten(), [1, 0, 1, 0, 0, 1, 0, 0, 0, 1]]
    res = mod.smooth(params)
    res_revised = res.apply(y_revised)
    news = res_revised.news(res, comparison_type='previous', tolerance=-1, revisions_details_start=revisions_details_start)
    data_revisions = news.data_revisions
    revision_details = news.revision_details_by_update.reset_index([2, 3])
    for key, diff in revisions.items():
        assert_allclose(data_revisions.loc[key, 'revised'], y_revised.loc[key])
        assert_allclose(data_revisions.loc[key, 'observed (prev)'], y.loc[key])
        assert_equal(np.array(data_revisions.loc[key, 'detailed impacts computed']), True)
        assert_allclose(revision_details.loc[key, 'revised'], y_revised.loc[key])
        assert_allclose(revision_details.loc[key, 'observed (prev)'], y.loc[key])
        assert_allclose(revision_details.loc[key, 'revision'], diff)
    key = ('2009Q3', 'realcons', '2009Q3', 'realcons')
    assert_allclose(revision_details.loc[key, 'weight'], 1)
    assert_allclose(revision_details.loc[key, 'impact'], revisions['2009Q3', 'realcons'])
    key = ('2009Q3', 'cpi', '2009Q3', 'cpi')
    assert_allclose(revision_details.loc[key, 'weight'], 1)
    assert_allclose(revision_details.loc[key, 'impact'], revisions['2009Q3', 'cpi'])
    key = ('2009Q2', 'realgdp', '2009Q3', 'realgdp')
    assert_allclose(revision_details.loc[key, 'weight'], ar_coeff['realgdp'])
    assert_allclose(revision_details.loc[key, 'impact'], ar_coeff['realgdp'] * revisions['2009Q2', 'realgdp'])
    key = ('2009Q2', 'realinv', '2009Q3', 'realinv')
    assert_allclose(revision_details.loc[key, 'weight'], 0)
    assert_allclose(revision_details.loc[key, 'impact'], 0)
    key = ('2009Q2', 'cpi', '2009Q3', 'cpi')
    assert_allclose(revision_details.loc[key, 'weight'], 0)
    assert_allclose(revision_details.loc[key, 'impact'], 0)
    assert_allclose(news.impacts['impact of news'], 0)
    assert_allclose(news.impacts['total impact'], news.impacts['impact of revisions'])
    for name in ['cpi', 'realcons', 'realinv']:
        assert_allclose(news.impacts.loc[('2009Q3', name), 'estimate (new)'], y_revised.loc['2009Q3', name])
        assert_allclose(news.impacts.loc[('2009Q3', name), 'estimate (prev)'], y.loc['2009Q3', name])
    name = 'realgdp'
    assert_allclose(news.impacts.loc[('2009Q3', name), 'estimate (new)'], y_revised.loc['2009Q2', name] * ar_coeff[name])
    assert_allclose(news.impacts.loc[('2009Q3', name), 'estimate (prev)'], y.loc['2009Q2', name] * ar_coeff[name])
    assert_allclose(news.impacts['impact of revisions'], revision_details.groupby(level=[2, 3]).sum()['impact'])