from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
def test_setitem_different_dtype(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), index=np.arange(5), columns=['c', 'b', 'a'])
    df.insert(0, 'foo', df['a'])
    df.insert(2, 'bar', df['c'])
    df['x'] = df['a'].astype('float32')
    result = df.dtypes
    expected = Series([np.dtype('float64')] * 5 + [np.dtype('float32')], index=['foo', 'c', 'bar', 'b', 'a', 'x'])
    tm.assert_series_equal(result, expected)
    df['a'] = df['a'].astype('float32')
    result = df.dtypes
    expected = Series([np.dtype('float64')] * 4 + [np.dtype('float32')] * 2, index=['foo', 'c', 'bar', 'b', 'a', 'x'])
    tm.assert_series_equal(result, expected)
    df['y'] = df['a'].astype('int32')
    result = df.dtypes
    expected = Series([np.dtype('float64')] * 4 + [np.dtype('float32')] * 2 + [np.dtype('int32')], index=['foo', 'c', 'bar', 'b', 'a', 'x', 'y'])
    tm.assert_series_equal(result, expected)