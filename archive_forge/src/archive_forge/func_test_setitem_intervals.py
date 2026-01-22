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
def test_setitem_intervals(self):
    df = DataFrame({'A': range(10)})
    ser = cut(df['A'], 5)
    assert isinstance(ser.cat.categories, IntervalIndex)
    df['B'] = ser
    df['C'] = np.array(ser)
    df['D'] = ser.values
    df['E'] = np.array(ser.values)
    df['F'] = ser.astype(object)
    assert isinstance(df['B'].dtype, CategoricalDtype)
    assert isinstance(df['B'].cat.categories.dtype, IntervalDtype)
    assert isinstance(df['D'].dtype, CategoricalDtype)
    assert isinstance(df['D'].cat.categories.dtype, IntervalDtype)
    assert isinstance(df['C'].dtype, IntervalDtype)
    assert isinstance(df['E'].dtype, IntervalDtype)
    assert is_object_dtype(df['F'])
    c = lambda x: Index(np.array(x))
    tm.assert_index_equal(c(df.B), c(df.B))
    tm.assert_index_equal(c(df.B), c(df.C), check_names=False)
    tm.assert_index_equal(c(df.B), c(df.D), check_names=False)
    tm.assert_index_equal(c(df.C), c(df.D), check_names=False)
    tm.assert_series_equal(df['B'], df['B'])
    tm.assert_series_equal(df['B'], df['D'], check_names=False)
    tm.assert_series_equal(df['C'], df['C'])
    tm.assert_series_equal(df['C'], df['E'], check_names=False)