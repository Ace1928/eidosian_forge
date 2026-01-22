import pytest
from pandas import (
import pandas._testing as tm
def test_groupby_sample_index_value_spans_groups():
    values = [1] * 3 + [2] * 3
    df = DataFrame({'a': values, 'b': values}, index=[1, 2, 2, 2, 2, 2])
    result = df.groupby('a').sample(n=2)
    values = [1] * 2 + [2] * 2
    expected = DataFrame({'a': values, 'b': values}, index=result.index)
    tm.assert_frame_equal(result, expected)
    result = df.groupby('a')['b'].sample(n=2)
    expected = Series(values, name='b', index=result.index)
    tm.assert_series_equal(result, expected)