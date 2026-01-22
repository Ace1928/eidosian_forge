from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
@pytest.mark.parametrize('revisions_details_start', [False, 202])
def test_grouped_revisions(revisions_details_start):
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
        assert_equal(np.array(data_revisions.loc[key, 'detailed impacts computed']), False)
    key = ('2009Q3', 'all prior revisions', '2009Q3')
    cols = ['revised', 'observed (prev)', 'revision', 'weight']
    assert np.all(revision_details.loc[key, cols].isnull())
    assert_allclose(revision_details.loc[key + ('realgdp',), 'impact'], ar_coeff['realgdp'] * revisions['2009Q2', 'realgdp'])
    assert_allclose(revision_details.loc[key + ('realcons',), 'impact'], revisions['2009Q3', 'realcons'])
    assert_allclose(revision_details.loc[key + ('realinv',), 'impact'], 0)
    assert_allclose(revision_details.loc[key + ('cpi',), 'impact'], revisions['2009Q3', 'cpi'])
    for name in ['cpi', 'realcons', 'realinv']:
        assert_allclose(news.impacts.loc[('2009Q3', name), 'estimate (new)'], y_revised.loc['2009Q3', name])
        assert_allclose(news.impacts.loc[('2009Q3', name), 'estimate (prev)'], y.loc['2009Q3', name])
    name = 'realgdp'
    assert_allclose(news.impacts.loc[('2009Q3', name), 'estimate (new)'], y_revised.loc['2009Q2', name] * ar_coeff[name])
    assert_allclose(news.impacts.loc[('2009Q3', name), 'estimate (prev)'], y.loc['2009Q2', name] * ar_coeff[name])
    assert_allclose(news.impacts['impact of revisions'], revision_details.groupby(level=[2, 3]).sum()['impact'])