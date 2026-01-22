from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_empty_string_nan_coerce_bug():
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = DataFrame({'a': [1, 1, 2, 2], 'b': ['', '', '', ''], 'c': pd.to_datetime([1, 2, 3, 4], unit='s')}).groupby(['a', 'b']).apply(lambda df: df.iloc[-1])
    expected = DataFrame([[1, '', pd.to_datetime(2, unit='s')], [2, '', pd.to_datetime(4, unit='s')]], columns=['a', 'b', 'c'], index=MultiIndex.from_tuples([(1, ''), (2, '')], names=['a', 'b']))
    tm.assert_frame_equal(result, expected)