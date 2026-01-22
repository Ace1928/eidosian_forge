import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('f', [lambda x: x.expanding(min_periods=5).cov(x, pairwise=True), lambda x: x.expanding(min_periods=5).corr(x, pairwise=True)])
def test_moment_functions_zero_length_pairwise(f):
    df1 = DataFrame()
    df2 = DataFrame(columns=Index(['a'], name='foo'), index=Index([], name='bar'))
    df2['a'] = df2['a'].astype('float64')
    df1_expected = DataFrame(index=MultiIndex.from_product([df1.index, df1.columns]))
    df2_expected = DataFrame(index=MultiIndex.from_product([df2.index, df2.columns], names=['bar', 'foo']), columns=Index(['a'], name='foo'), dtype='float64')
    df1_result = f(df1)
    tm.assert_frame_equal(df1_result, df1_expected)
    df2_result = f(df2)
    tm.assert_frame_equal(df2_result, df2_expected)