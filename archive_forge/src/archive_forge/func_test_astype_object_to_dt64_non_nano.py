from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', [None, 'UTC', 'US/Pacific'])
def test_astype_object_to_dt64_non_nano(self, tz):
    ts = Timestamp('2999-01-01')
    dtype = 'M8[us]'
    if tz is not None:
        dtype = f'M8[us, {tz}]'
    vals = [ts, '2999-01-02 03:04:05.678910', 2500]
    ser = Series(vals, dtype=object)
    result = ser.astype(dtype)
    pointwise = [vals[0].tz_localize(tz), Timestamp(vals[1], tz=tz), to_datetime(vals[2], unit='us', utc=True).tz_convert(tz)]
    exp_vals = [x.as_unit('us').asm8 for x in pointwise]
    exp_arr = np.array(exp_vals, dtype='M8[us]')
    expected = Series(exp_arr, dtype='M8[us]')
    if tz is not None:
        expected = expected.dt.tz_localize('UTC').dt.tz_convert(tz)
    tm.assert_series_equal(result, expected)