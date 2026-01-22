from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('test_series', [True, False])
def test_apply_with_duplicated_non_sorted_axis(test_series):
    df = DataFrame([['x', 'p'], ['x', 'p'], ['x', 'o']], columns=['X', 'Y'], index=[1, 2, 2])
    if test_series:
        ser = df.set_index('Y')['X']
        result = ser.groupby(level=0, group_keys=False).apply(lambda x: x)
        result = result.sort_index()
        expected = ser.sort_index()
        tm.assert_series_equal(result, expected)
    else:
        msg = 'DataFrameGroupBy.apply operated on the grouping columns'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            result = df.groupby('Y', group_keys=False).apply(lambda x: x)
        result = result.sort_values('Y')
        expected = df.sort_values('Y')
        tm.assert_frame_equal(result, expected)