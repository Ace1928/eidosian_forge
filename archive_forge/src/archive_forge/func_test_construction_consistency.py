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
def test_construction_consistency(self):
    ser = Series(date_range('20130101', periods=3, tz='US/Eastern'))
    result = Series(ser, dtype=ser.dtype)
    tm.assert_series_equal(result, ser)
    result = Series(ser.dt.tz_convert('UTC'), dtype=ser.dtype)
    tm.assert_series_equal(result, ser)
    result = Series(ser.values, dtype=ser.dtype)
    expected = Series(ser.values).dt.tz_localize(ser.dtype.tz)
    tm.assert_series_equal(result, expected)
    with tm.assert_produces_warning(None):
        middle = Series(ser.values).dt.tz_localize('UTC')
        result = middle.dt.tz_convert(ser.dtype.tz)
    tm.assert_series_equal(result, ser)
    with tm.assert_produces_warning(None):
        result = Series(ser.values.view('int64'), dtype=ser.dtype)
    tm.assert_series_equal(result, ser)