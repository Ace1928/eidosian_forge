from datetime import (
import pytz
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
import pandas.util._test_decorators as td
from pandas import Timestamp
import pandas._testing as tm
def test_to_pydatetime_bijective(self):
    exp_warning = None if Timestamp.max.nanosecond == 0 else UserWarning
    with tm.assert_produces_warning(exp_warning):
        pydt_max = Timestamp.max.to_pydatetime()
    assert Timestamp(pydt_max).as_unit('ns')._value / 1000 == Timestamp.max._value / 1000
    exp_warning = None if Timestamp.min.nanosecond == 0 else UserWarning
    with tm.assert_produces_warning(exp_warning):
        pydt_min = Timestamp.min.to_pydatetime()
    tdus = timedelta(microseconds=1)
    assert pydt_min + tdus > Timestamp.min
    assert Timestamp(pydt_min + tdus).as_unit('ns')._value / 1000 == Timestamp.min._value / 1000