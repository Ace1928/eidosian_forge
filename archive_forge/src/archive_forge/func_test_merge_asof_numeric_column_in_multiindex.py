import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_merge_asof_numeric_column_in_multiindex():
    left = pd.DataFrame({'b': [10, 11, 12]}, index=pd.MultiIndex.from_arrays([[1, 2, 3], ['a', 'b', 'c']], names=['a', 'z']))
    right = pd.DataFrame({'c': [20, 21, 22]}, index=pd.MultiIndex.from_arrays([[1, 2, 3], ['x', 'y', 'z']], names=['a', 'y']))
    result = merge_asof(left, right, left_on='a', right_on='a')
    expected = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 11, 12], 'c': [20, 21, 22]})
    tm.assert_frame_equal(result, expected)