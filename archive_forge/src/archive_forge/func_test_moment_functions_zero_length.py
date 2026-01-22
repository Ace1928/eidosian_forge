import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('f', [lambda x: x.expanding().count(), lambda x: x.expanding(min_periods=5).cov(x, pairwise=False), lambda x: x.expanding(min_periods=5).corr(x, pairwise=False), lambda x: x.expanding(min_periods=5).max(), lambda x: x.expanding(min_periods=5).min(), lambda x: x.expanding(min_periods=5).sum(), lambda x: x.expanding(min_periods=5).mean(), lambda x: x.expanding(min_periods=5).std(), lambda x: x.expanding(min_periods=5).var(), lambda x: x.expanding(min_periods=5).skew(), lambda x: x.expanding(min_periods=5).kurt(), lambda x: x.expanding(min_periods=5).quantile(0.5), lambda x: x.expanding(min_periods=5).median(), lambda x: x.expanding(min_periods=5).apply(sum, raw=False), lambda x: x.expanding(min_periods=5).apply(sum, raw=True)])
def test_moment_functions_zero_length(f):
    s = Series(dtype=np.float64)
    s_expected = s
    df1 = DataFrame()
    df1_expected = df1
    df2 = DataFrame(columns=['a'])
    df2['a'] = df2['a'].astype('float64')
    df2_expected = df2
    s_result = f(s)
    tm.assert_series_equal(s_result, s_expected)
    df1_result = f(df1)
    tm.assert_frame_equal(df1_result, df1_expected)
    df2_result = f(df2)
    tm.assert_frame_equal(df2_result, df2_expected)