from datetime import datetime
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
def test_tz_convert_unsorted(self, tzstr):
    dr = date_range('2012-03-09', freq='h', periods=100, tz='utc')
    dr = dr.tz_convert(tzstr)
    result = dr[::-1].hour
    exp = dr.hour[::-1]
    tm.assert_almost_equal(result, exp)