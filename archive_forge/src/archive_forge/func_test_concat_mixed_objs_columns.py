from collections import (
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tests.extension.decimal import to_decimal
def test_concat_mixed_objs_columns(self):
    index = date_range('01-Jan-2013', periods=10, freq='h')
    arr = np.arange(10, dtype='int64')
    s1 = Series(arr, index=index)
    s2 = Series(arr, index=index)
    df = DataFrame(arr.reshape(-1, 1), index=index)
    expected = DataFrame(np.repeat(arr, 2).reshape(-1, 2), index=index, columns=[0, 0])
    result = concat([df, df], axis=1)
    tm.assert_frame_equal(result, expected)
    expected = DataFrame(np.repeat(arr, 2).reshape(-1, 2), index=index, columns=[0, 1])
    result = concat([s1, s2], axis=1)
    tm.assert_frame_equal(result, expected)
    expected = DataFrame(np.repeat(arr, 3).reshape(-1, 3), index=index, columns=[0, 1, 2])
    result = concat([s1, s2, s1], axis=1)
    tm.assert_frame_equal(result, expected)
    expected = DataFrame(np.repeat(arr, 5).reshape(-1, 5), index=index, columns=[0, 0, 1, 2, 3])
    result = concat([s1, df, s2, s2, s1], axis=1)
    tm.assert_frame_equal(result, expected)
    s1.name = 'foo'
    expected = DataFrame(np.repeat(arr, 3).reshape(-1, 3), index=index, columns=['foo', 0, 0])
    result = concat([s1, df, s2], axis=1)
    tm.assert_frame_equal(result, expected)
    s2.name = 'bar'
    expected = DataFrame(np.repeat(arr, 3).reshape(-1, 3), index=index, columns=['foo', 0, 'bar'])
    result = concat([s1, df, s2], axis=1)
    tm.assert_frame_equal(result, expected)
    expected = DataFrame(np.repeat(arr, 3).reshape(-1, 3), index=index, columns=[0, 1, 2])
    result = concat([s1, df, s2], axis=1, ignore_index=True)
    tm.assert_frame_equal(result, expected)