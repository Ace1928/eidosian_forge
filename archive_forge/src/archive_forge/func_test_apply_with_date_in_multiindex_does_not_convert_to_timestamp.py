from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_with_date_in_multiindex_does_not_convert_to_timestamp():
    df = DataFrame({'A': ['a', 'a', 'a', 'b'], 'B': [date(2020, 1, 10), date(2020, 1, 10), date(2020, 2, 10), date(2020, 2, 10)], 'C': [1, 2, 3, 4]}, index=Index([100, 101, 102, 103], name='idx'))
    grp = df.groupby(['A', 'B'])
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = grp.apply(lambda x: x.head(1))
    expected = df.iloc[[0, 2, 3]]
    expected = expected.reset_index()
    expected.index = MultiIndex.from_frame(expected[['A', 'B', 'idx']])
    expected = expected.drop(columns='idx')
    tm.assert_frame_equal(result, expected)
    for val in result.index.levels[1]:
        assert type(val) is date