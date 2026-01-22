import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('by', [[0, 0, 0, 0], [0, 1, 1, 1], [1, 0, 1, 1], [0, None, None, None], pytest.param([None, None, None, None], marks=pytest.mark.xfail)])
def test_size_axis_1(df, axis_1, by, sort, dropna):
    counts = {key: sum((value == key for value in by)) for key in dict.fromkeys(by)}
    if dropna:
        counts = {key: value for key, value in counts.items() if key is not None}
    expected = Series(counts, dtype='int64')
    if sort:
        expected = expected.sort_index()
    if tm.is_integer_dtype(expected.index) and (not any((x is None for x in by))):
        expected.index = expected.index.astype(np.int_)
    grouped = df.groupby(by=by, axis=axis_1, sort=sort, dropna=dropna)
    result = grouped.size()
    tm.assert_series_equal(result, expected)