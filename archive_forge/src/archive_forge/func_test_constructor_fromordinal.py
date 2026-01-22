import calendar
from datetime import (
import zoneinfo
import dateutil.tz
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.compat import PY310
from pandas.errors import OutOfBoundsDatetime
from pandas import (
def test_constructor_fromordinal(self):
    base = datetime(2000, 1, 1)
    ts = Timestamp.fromordinal(base.toordinal())
    assert base == ts
    assert base.toordinal() == ts.toordinal()
    ts = Timestamp.fromordinal(base.toordinal(), tz='US/Eastern')
    assert Timestamp('2000-01-01', tz='US/Eastern') == ts
    assert base.toordinal() == ts.toordinal()
    dt = datetime(2011, 4, 16, 0, 0)
    ts = Timestamp.fromordinal(dt.toordinal())
    assert ts.to_pydatetime() == dt
    stamp = Timestamp('2011-4-16', tz='US/Eastern')
    dt_tz = stamp.to_pydatetime()
    ts = Timestamp.fromordinal(dt_tz.toordinal(), tz='US/Eastern')
    assert ts.to_pydatetime() == dt_tz