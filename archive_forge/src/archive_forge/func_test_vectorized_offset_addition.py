from __future__ import annotations
from datetime import datetime
import pytest
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
from pandas import (
from pandas.tests.tseries.offsets.common import (
@pytest.mark.parametrize('klass', [Series, DatetimeIndex])
def test_vectorized_offset_addition(self, klass):
    shift = klass([Timestamp('2000-01-15 00:15:00', tz='US/Central'), Timestamp('2000-02-15', tz='US/Central')], name='a')
    with tm.assert_produces_warning(None):
        result = shift + SemiMonthBegin()
        result2 = SemiMonthBegin() + shift
    exp = klass([Timestamp('2000-02-01 00:15:00', tz='US/Central'), Timestamp('2000-03-01', tz='US/Central')], name='a')
    tm.assert_equal(result, exp)
    tm.assert_equal(result2, exp)
    shift = klass([Timestamp('2000-01-01 00:15:00', tz='US/Central'), Timestamp('2000-02-01', tz='US/Central')], name='a')
    with tm.assert_produces_warning(None):
        result = shift + SemiMonthBegin()
        result2 = SemiMonthBegin() + shift
    exp = klass([Timestamp('2000-01-15 00:15:00', tz='US/Central'), Timestamp('2000-02-15', tz='US/Central')], name='a')
    tm.assert_equal(result, exp)
    tm.assert_equal(result2, exp)