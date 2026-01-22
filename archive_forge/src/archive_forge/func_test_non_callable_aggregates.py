from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('how', ['agg', 'apply'])
def test_non_callable_aggregates(how):
    df = DataFrame({'A': [None, 2, 3], 'B': [1.0, np.nan, 3.0], 'C': ['foo', None, 'bar']})
    result = getattr(df, how)({'A': 'count'})
    expected = Series({'A': 2})
    tm.assert_series_equal(result, expected)
    result = getattr(df, how)({'A': 'size'})
    expected = Series({'A': 3})
    tm.assert_series_equal(result, expected)
    result1 = getattr(df, how)(['count', 'size'])
    result2 = getattr(df, how)({'A': ['count', 'size'], 'B': ['count', 'size'], 'C': ['count', 'size']})
    expected = DataFrame({'A': {'count': 2, 'size': 3}, 'B': {'count': 2, 'size': 3}, 'C': {'count': 2, 'size': 3}})
    tm.assert_frame_equal(result1, result2, check_like=True)
    tm.assert_frame_equal(result2, expected, check_like=True)
    result = getattr(df, how)('count')
    expected = df.count()
    tm.assert_series_equal(result, expected)