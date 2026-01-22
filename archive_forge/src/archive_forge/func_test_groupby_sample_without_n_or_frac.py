import pytest
from pandas import (
import pandas._testing as tm
def test_groupby_sample_without_n_or_frac():
    values = [1] * 10 + [2] * 10
    df = DataFrame({'a': values, 'b': values})
    result = df.groupby('a').sample(n=None, frac=None)
    expected = DataFrame({'a': [1, 2], 'b': [1, 2]}, index=result.index)
    tm.assert_frame_equal(result, expected)
    result = df.groupby('a')['b'].sample(n=None, frac=None)
    expected = Series([1, 2], name='b', index=result.index)
    tm.assert_series_equal(result, expected)