from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_handle_join_key_pass_array(self):
    left = DataFrame({'key': [1, 1, 2, 2, 3], 'value': np.arange(5)}, columns=['value', 'key'], dtype='int64')
    right = DataFrame({'rvalue': np.arange(6)}, dtype='int64')
    key = np.array([1, 1, 2, 3, 4, 5], dtype='int64')
    merged = merge(left, right, left_on='key', right_on=key, how='outer')
    merged2 = merge(right, left, left_on=key, right_on='key', how='outer')
    tm.assert_series_equal(merged['key'], merged2['key'])
    assert merged['key'].notna().all()
    assert merged2['key'].notna().all()
    left = DataFrame({'value': np.arange(5)}, columns=['value'])
    right = DataFrame({'rvalue': np.arange(6)})
    lkey = np.array([1, 1, 2, 2, 3])
    rkey = np.array([1, 1, 2, 3, 4, 5])
    merged = merge(left, right, left_on=lkey, right_on=rkey, how='outer')
    expected = Series([1, 1, 1, 1, 2, 2, 3, 4, 5], dtype=int, name='key_0')
    tm.assert_series_equal(merged['key_0'], expected)
    left = DataFrame({'value': np.arange(3)})
    right = DataFrame({'rvalue': np.arange(6)})
    key = np.array([0, 1, 1, 2, 2, 3], dtype=np.int64)
    merged = merge(left, right, left_index=True, right_on=key, how='outer')
    tm.assert_series_equal(merged['key_0'], Series(key, name='key_0'))