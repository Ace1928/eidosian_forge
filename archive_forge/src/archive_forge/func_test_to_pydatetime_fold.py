from datetime import (
import pytz
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
import pandas.util._test_decorators as td
from pandas import Timestamp
import pandas._testing as tm
def test_to_pydatetime_fold(self):
    tzstr = 'dateutil/usr/share/zoneinfo/America/Chicago'
    ts = Timestamp(year=2013, month=11, day=3, hour=1, minute=0, fold=1, tz=tzstr)
    dt = ts.to_pydatetime()
    assert dt.fold == 1