from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_sort_index_groups():
    df = DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 0], 'C': [1, 1, 1, 2, 2]}, index=range(5))
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('C').apply(lambda x: x.A.sort_index())
    expected = Series(range(1, 6), index=MultiIndex.from_tuples([(1, 0), (1, 1), (1, 2), (2, 3), (2, 4)], names=['C', None]), name='A')
    tm.assert_series_equal(result, expected)