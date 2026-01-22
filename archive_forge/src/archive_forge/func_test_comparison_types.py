from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
def test_comparison_types():
    endog = dta['infl'].copy()
    endog.iloc[-1] = np.nan
    msg = 'Could not automatically determine the type of comparison'
    mod = sarimax.SARIMAX(endog)
    res = mod.smooth([0.5, 1.0])
    with pytest.raises(ValueError, match=msg):
        res.news(endog)
    with pytest.raises(ValueError, match=msg):
        res.news(res)
    news = res.news(endog, comparison_type='previous')
    assert_allclose(news.total_impacts, 0)
    news = res.news(endog, comparison_type='updated')
    assert_allclose(news.total_impacts, 0)
    news = res.news(res, comparison_type='updated')
    assert_allclose(news.total_impacts, 0)
    news = res.news(res, comparison_type='updated')
    assert_allclose(news.total_impacts, 0)