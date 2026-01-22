import calendar
from datetime import (
import locale
import unicodedata
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas.errors import SettingWithCopyError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('freq', ['D', 's', 'ms'])
def test_dt_namespace_accessor_datetime64(self, freq):
    dti = date_range('20130101', periods=5, freq=freq)
    ser = Series(dti, name='xxx')
    for prop in ok_for_dt:
        if prop != 'freq':
            self._compare(ser, prop)
    for prop in ok_for_dt_methods:
        getattr(ser.dt, prop)
    msg = 'The behavior of DatetimeProperties.to_pydatetime is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = ser.dt.to_pydatetime()
    assert isinstance(result, np.ndarray)
    assert result.dtype == object
    result = ser.dt.tz_localize('US/Eastern')
    exp_values = DatetimeIndex(ser.values).tz_localize('US/Eastern')
    expected = Series(exp_values, index=ser.index, name='xxx')
    tm.assert_series_equal(result, expected)
    tz_result = result.dt.tz
    assert str(tz_result) == 'US/Eastern'
    freq_result = ser.dt.freq
    assert freq_result == DatetimeIndex(ser.values, freq='infer').freq
    result = ser.dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    exp_values = DatetimeIndex(ser.values).tz_localize('UTC').tz_convert('US/Eastern')
    expected = Series(exp_values, index=ser.index, name='xxx')
    tm.assert_series_equal(result, expected)