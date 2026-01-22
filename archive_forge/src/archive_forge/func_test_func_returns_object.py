from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_func_returns_object():
    df = DataFrame({'a': [1, 2]}, index=Index([1, 2]))
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('a').apply(lambda g: g.index)
    expected = Series([Index([1]), Index([2])], index=Index([1, 2], name='a'))
    tm.assert_series_equal(result, expected)