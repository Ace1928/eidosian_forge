import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
@pytest.mark.parametrize('f', [lambda x: x.rolling(window=10, min_periods=5).cov(x, pairwise=True), lambda x: x.rolling(window=10, min_periods=5).corr(x, pairwise=True)])
def test_rolling_functions_window_non_shrinkage_binary(f):
    df = DataFrame([[1, 5], [3, 2], [3, 9], [-1, 0]], columns=Index(['A', 'B'], name='foo'), index=Index(range(4), name='bar'))
    df_expected = DataFrame(columns=Index(['A', 'B'], name='foo'), index=MultiIndex.from_product([df.index, df.columns], names=['bar', 'foo']), dtype='float64')
    df_result = f(df)
    tm.assert_frame_equal(df_result, df_expected)