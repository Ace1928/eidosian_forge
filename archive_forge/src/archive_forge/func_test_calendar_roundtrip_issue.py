import contextlib
import datetime as dt
import hashlib
import tempfile
import time
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import (
def test_calendar_roundtrip_issue(setup_path):
    weekmask_egypt = 'Sun Mon Tue Wed Thu'
    holidays = ['2012-05-01', dt.datetime(2013, 5, 1), np.datetime64('2014-05-01')]
    bday_egypt = pd.offsets.CustomBusinessDay(holidays=holidays, weekmask=weekmask_egypt)
    mydt = dt.datetime(2013, 4, 30)
    dts = date_range(mydt, periods=5, freq=bday_egypt)
    s = Series(dts.weekday, dts).map(Series('Mon Tue Wed Thu Fri Sat Sun'.split()))
    with ensure_clean_store(setup_path) as store:
        store.put('fixed', s)
        result = store.select('fixed')
        tm.assert_series_equal(result, s)
        store.append('table', s)
        result = store.select('table')
        tm.assert_series_equal(result, s)