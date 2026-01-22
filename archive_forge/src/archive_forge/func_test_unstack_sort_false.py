from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
@pytest.mark.parametrize('dtype', ['float64', 'Float64'])
def test_unstack_sort_false(frame_or_series, dtype):
    index = MultiIndex.from_tuples([('two', 'z', 'b'), ('two', 'y', 'a'), ('one', 'z', 'b'), ('one', 'y', 'a')])
    obj = frame_or_series(np.arange(1.0, 5.0), index=index, dtype=dtype)
    result = obj.unstack(level=-1, sort=False)
    if frame_or_series is DataFrame:
        expected_columns = MultiIndex.from_tuples([(0, 'b'), (0, 'a')])
    else:
        expected_columns = ['b', 'a']
    expected = DataFrame([[1.0, np.nan], [np.nan, 2.0], [3.0, np.nan], [np.nan, 4.0]], columns=expected_columns, index=MultiIndex.from_tuples([('two', 'z'), ('two', 'y'), ('one', 'z'), ('one', 'y')]), dtype=dtype)
    tm.assert_frame_equal(result, expected)
    result = obj.unstack(level=[1, 2], sort=False)
    if frame_or_series is DataFrame:
        expected_columns = MultiIndex.from_tuples([(0, 'z', 'b'), (0, 'y', 'a')])
    else:
        expected_columns = MultiIndex.from_tuples([('z', 'b'), ('y', 'a')])
    expected = DataFrame([[1.0, 2.0], [3.0, 4.0]], index=['two', 'one'], columns=expected_columns, dtype=dtype)
    tm.assert_frame_equal(result, expected)