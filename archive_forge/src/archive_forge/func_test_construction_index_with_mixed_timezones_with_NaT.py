from __future__ import annotations
from datetime import (
from functools import partial
from operator import attrgetter
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_construction_index_with_mixed_timezones_with_NaT(self):
    result = Index([pd.NaT, Timestamp('2011-01-01'), pd.NaT, Timestamp('2011-01-02')], name='idx')
    exp = DatetimeIndex([pd.NaT, Timestamp('2011-01-01'), pd.NaT, Timestamp('2011-01-02')], name='idx')
    tm.assert_index_equal(result, exp, exact=True)
    assert isinstance(result, DatetimeIndex)
    assert result.tz is None
    result = Index([pd.NaT, Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), pd.NaT, Timestamp('2011-01-02 10:00', tz='Asia/Tokyo')], name='idx')
    exp = DatetimeIndex([pd.NaT, Timestamp('2011-01-01 10:00'), pd.NaT, Timestamp('2011-01-02 10:00')], tz='Asia/Tokyo', name='idx')
    tm.assert_index_equal(result, exp, exact=True)
    assert isinstance(result, DatetimeIndex)
    assert result.tz is not None
    assert result.tz == exp.tz
    result = Index([Timestamp('2011-01-01 10:00', tz='US/Eastern'), pd.NaT, Timestamp('2011-08-01 10:00', tz='US/Eastern')], name='idx')
    exp = DatetimeIndex([Timestamp('2011-01-01 10:00'), pd.NaT, Timestamp('2011-08-01 10:00')], tz='US/Eastern', name='idx')
    tm.assert_index_equal(result, exp, exact=True)
    assert isinstance(result, DatetimeIndex)
    assert result.tz is not None
    assert result.tz == exp.tz
    result = Index([pd.NaT, Timestamp('2011-01-01 10:00'), pd.NaT, Timestamp('2011-01-02 10:00', tz='US/Eastern')], name='idx')
    exp = Index([pd.NaT, Timestamp('2011-01-01 10:00'), pd.NaT, Timestamp('2011-01-02 10:00', tz='US/Eastern')], dtype='object', name='idx')
    tm.assert_index_equal(result, exp, exact=True)
    assert not isinstance(result, DatetimeIndex)
    result = Index([pd.NaT, Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), pd.NaT, Timestamp('2011-01-02 10:00', tz='US/Eastern')], name='idx')
    exp = Index([pd.NaT, Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), pd.NaT, Timestamp('2011-01-02 10:00', tz='US/Eastern')], dtype='object', name='idx')
    tm.assert_index_equal(result, exp, exact=True)
    assert not isinstance(result, DatetimeIndex)
    result = Index([pd.NaT, pd.NaT], name='idx')
    exp = DatetimeIndex([pd.NaT, pd.NaT], name='idx')
    tm.assert_index_equal(result, exp, exact=True)
    assert isinstance(result, DatetimeIndex)
    assert result.tz is None