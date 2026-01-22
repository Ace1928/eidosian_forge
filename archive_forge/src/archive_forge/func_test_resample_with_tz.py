from datetime import datetime
import warnings
import dateutil
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import _get_period_range_edges
from pandas.tseries import offsets
@pytest.mark.parametrize('tz', [pytz.timezone('America/Los_Angeles'), dateutil.tz.gettz('America/Los_Angeles')])
def test_resample_with_tz(self, tz, unit):
    dti = date_range('2017-01-01', periods=48, freq='h', tz=tz, unit=unit)
    ser = Series(2, index=dti)
    result = ser.resample('D').mean()
    exp_dti = pd.DatetimeIndex(['2017-01-01', '2017-01-02'], tz=tz, freq='D').as_unit(unit)
    expected = Series(2.0, index=exp_dti)
    tm.assert_series_equal(result, expected)
    assert result.index.tz == tz