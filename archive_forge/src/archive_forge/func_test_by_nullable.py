import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_by_nullable(self, any_numeric_ea_dtype, using_infer_string):
    arr = pd.array([pd.NA, 0, 1], dtype=any_numeric_ea_dtype)
    if arr.dtype.kind in ['i', 'u']:
        max_val = np.iinfo(arr.dtype.numpy_dtype).max
    else:
        max_val = np.finfo(arr.dtype.numpy_dtype).max
    arr[2] = max_val
    left = pd.DataFrame({'by_col1': arr, 'by_col2': ['HELLO', 'To', 'You'], 'on_col': [2, 4, 6], 'value': ['a', 'c', 'e']})
    right = pd.DataFrame({'by_col1': arr, 'by_col2': ['WORLD', 'Wide', 'Web'], 'on_col': [1, 2, 6], 'value': ['b', 'd', 'f']})
    result = merge_asof(left, right, by=['by_col1', 'by_col2'], on='on_col')
    expected = pd.DataFrame({'by_col1': arr, 'by_col2': ['HELLO', 'To', 'You'], 'on_col': [2, 4, 6], 'value_x': ['a', 'c', 'e']})
    expected['value_y'] = np.array([np.nan, np.nan, np.nan], dtype=object)
    if using_infer_string:
        expected['value_y'] = expected['value_y'].astype('string[pyarrow_numpy]')
    tm.assert_frame_equal(result, expected)