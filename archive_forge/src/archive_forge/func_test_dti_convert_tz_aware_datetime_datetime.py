from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', [pytz.timezone('US/Eastern'), gettz('US/Eastern')])
def test_dti_convert_tz_aware_datetime_datetime(self, tz):
    dates = [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)]
    dates_aware = [conversion.localize_pydatetime(x, tz) for x in dates]
    result = DatetimeIndex(dates_aware).as_unit('ns')
    assert timezones.tz_compare(result.tz, tz)
    converted = to_datetime(dates_aware, utc=True).as_unit('ns')
    ex_vals = np.array([Timestamp(x).as_unit('ns')._value for x in dates_aware])
    tm.assert_numpy_array_equal(converted.asi8, ex_vals)
    assert converted.tz is timezone.utc