import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
@pytest.mark.parametrize('bins', [3, [Timestamp('2013-01-01 04:57:07.200000', tz='UTC').tz_convert('US/Eastern'), Timestamp('2013-01-01 21:00:00', tz='UTC').tz_convert('US/Eastern'), Timestamp('2013-01-02 13:00:00', tz='UTC').tz_convert('US/Eastern'), Timestamp('2013-01-03 05:00:00', tz='UTC').tz_convert('US/Eastern')]])
@pytest.mark.parametrize('box', [list, np.array, Index, Series])
def test_datetime_tz_cut(bins, box):
    tz = 'US/Eastern'
    ser = Series(date_range('20130101', periods=3, tz=tz))
    if not isinstance(bins, int):
        bins = box(bins)
    result = cut(ser, bins)
    expected = Series(IntervalIndex([Interval(Timestamp('2012-12-31 23:57:07.200000', tz=tz), Timestamp('2013-01-01 16:00:00', tz=tz)), Interval(Timestamp('2013-01-01 16:00:00', tz=tz), Timestamp('2013-01-02 08:00:00', tz=tz)), Interval(Timestamp('2013-01-02 08:00:00', tz=tz), Timestamp('2013-01-03 00:00:00', tz=tz))])).astype(CategoricalDtype(ordered=True))
    tm.assert_series_equal(result, expected)