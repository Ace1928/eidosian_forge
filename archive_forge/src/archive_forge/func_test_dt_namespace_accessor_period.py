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
def test_dt_namespace_accessor_period(self):
    pi = period_range('20130101', periods=5, freq='D')
    ser = Series(pi, name='xxx')
    for prop in ok_for_period:
        if prop != 'freq':
            self._compare(ser, prop)
    for prop in ok_for_period_methods:
        getattr(ser.dt, prop)
    freq_result = ser.dt.freq
    assert freq_result == PeriodIndex(ser.values).freq