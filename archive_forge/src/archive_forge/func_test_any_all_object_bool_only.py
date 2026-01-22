from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
def test_any_all_object_bool_only(self):
    df = DataFrame({'A': ['foo', 2], 'B': [True, False]}).astype(object)
    df._consolidate_inplace()
    df['C'] = Series([True, True])
    df['D'] = df['C'].astype('category')
    res = df._get_bool_data()
    expected = df[['C']]
    tm.assert_frame_equal(res, expected)
    res = df.all(bool_only=True, axis=0)
    expected = Series([True], index=['C'])
    tm.assert_series_equal(res, expected)
    res = df[['B', 'C']].all(bool_only=True, axis=0)
    tm.assert_series_equal(res, expected)
    assert df.all(bool_only=True, axis=None)
    res = df.any(bool_only=True, axis=0)
    expected = Series([True], index=['C'])
    tm.assert_series_equal(res, expected)
    res = df[['C']].any(bool_only=True, axis=0)
    tm.assert_series_equal(res, expected)
    assert df.any(bool_only=True, axis=None)