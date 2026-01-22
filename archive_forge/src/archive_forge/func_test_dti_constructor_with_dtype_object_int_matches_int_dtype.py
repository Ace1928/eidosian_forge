from __future__ import annotations
from datetime import (
from functools import partial
from operator import attrgetter
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
@pytest.mark.parametrize('dtype', ['M8[us]', 'M8[us, US/Pacific]'])
def test_dti_constructor_with_dtype_object_int_matches_int_dtype(self, dtype):
    vals1 = np.arange(5, dtype='i8') * 1000
    vals1[0] = pd.NaT.value
    vals2 = vals1.astype(np.float64)
    vals2[0] = np.nan
    vals3 = vals1.astype(object)
    vals3[0] = pd.NaT
    vals4 = vals2.astype(object)
    res1 = DatetimeIndex(vals1, dtype=dtype)
    res2 = DatetimeIndex(vals2, dtype=dtype)
    res3 = DatetimeIndex(vals3, dtype=dtype)
    res4 = DatetimeIndex(vals4, dtype=dtype)
    expected = DatetimeIndex(vals1.view('M8[us]'))
    if res1.tz is not None:
        expected = expected.tz_localize('UTC').tz_convert(res1.tz)
    tm.assert_index_equal(res1, expected)
    tm.assert_index_equal(res2, expected)
    tm.assert_index_equal(res3, expected)
    tm.assert_index_equal(res4, expected)