from datetime import time
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
def test_at_time_tz(self):
    dti = date_range('2018', periods=3, freq='h', tz='US/Pacific')
    df = DataFrame(list(range(len(dti))), index=dti)
    result = df.at_time(time(4, tzinfo=pytz.timezone('US/Eastern')))
    expected = df.iloc[1:2]
    tm.assert_frame_equal(result, expected)