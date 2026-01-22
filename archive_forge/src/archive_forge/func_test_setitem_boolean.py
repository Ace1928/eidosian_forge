from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setitem_boolean(self, float_frame):
    df = float_frame.copy()
    values = float_frame.values.copy()
    df[df['A'] > 0] = 4
    values[values[:, 0] > 0] = 4
    tm.assert_almost_equal(df.values, values)
    series = df['A'] == 4
    series = series.reindex(df.index[::-1])
    df[series] = 1
    values[values[:, 0] == 4] = 1
    tm.assert_almost_equal(df.values, values)
    df[df > 0] = 5
    values[values > 0] = 5
    tm.assert_almost_equal(df.values, values)
    df[df == 5] = 0
    values[values == 5] = 0
    tm.assert_almost_equal(df.values, values)
    df[df[:-1] < 0] = 2
    np.putmask(values[:-1], values[:-1] < 0, 2)
    tm.assert_almost_equal(df.values, values)
    df[df[::-1] == 2] = 3
    values[values == 2] = 3
    tm.assert_almost_equal(df.values, values)
    msg = 'Must pass DataFrame or 2-d ndarray with boolean values only'
    with pytest.raises(TypeError, match=msg):
        df[df * 0] = 2
    df_orig = df.copy()
    mask = df > np.abs(df)
    df[df > np.abs(df)] = np.nan
    values = df_orig.values.copy()
    values[mask.values] = np.nan
    expected = DataFrame(values, index=df_orig.index, columns=df_orig.columns)
    tm.assert_frame_equal(df, expected)
    df[df > np.abs(df)] = df * 2
    np.putmask(values, mask.values, df.values * 2)
    expected = DataFrame(values, index=df_orig.index, columns=df_orig.columns)
    tm.assert_frame_equal(df, expected)