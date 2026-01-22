from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
def test_constructor_dtype_datetime64_7(self):
    dates = date_range('01-Jan-2015', '01-Dec-2015', freq='ME')
    values2 = dates.view(np.ndarray).astype('datetime64[ns]')
    expected = Series(values2, index=dates)
    for unit in ['s', 'D', 'ms', 'us', 'ns']:
        dtype = np.dtype(f'M8[{unit}]')
        values1 = dates.view(np.ndarray).astype(dtype)
        result = Series(values1, dates)
        if unit == 'D':
            dtype = np.dtype('M8[s]')
        assert result.dtype == dtype
        tm.assert_series_equal(result, expected.astype(dtype))
    expected = Series(values2, index=dates, dtype=object)
    for dtype in ['s', 'D', 'ms', 'us', 'ns']:
        values1 = dates.view(np.ndarray).astype(f'M8[{dtype}]')
        result = Series(values1, index=dates, dtype=object)
        tm.assert_series_equal(result, expected)
    dates2 = np.array([d.date() for d in dates.to_pydatetime()], dtype=object)
    series1 = Series(dates2, dates)
    tm.assert_numpy_array_equal(series1.values, dates2)
    assert series1.dtype == object