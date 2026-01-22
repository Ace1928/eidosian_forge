from datetime import (
import itertools
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock
def test_construction_with_mixed(self, float_string_frame, using_infer_string):
    data = [[datetime(2001, 1, 5), np.nan, datetime(2001, 1, 2)], [datetime(2000, 1, 2), datetime(2000, 1, 3), datetime(2000, 1, 1)]]
    df = DataFrame(data)
    result = df.dtypes
    expected = Series({'datetime64[us]': 3})
    float_string_frame['datetime'] = datetime.now()
    float_string_frame['timedelta'] = timedelta(days=1, seconds=1)
    assert float_string_frame['datetime'].dtype == 'M8[us]'
    assert float_string_frame['timedelta'].dtype == 'm8[us]'
    result = float_string_frame.dtypes
    expected = Series([np.dtype('float64')] * 4 + [np.dtype('object') if not using_infer_string else 'string', np.dtype('datetime64[us]'), np.dtype('timedelta64[us]')], index=list('ABCD') + ['foo', 'datetime', 'timedelta'])
    tm.assert_series_equal(result, expected)