from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
def test_insert4(self, unit):
    for tz in ['US/Pacific', 'Asia/Singapore']:
        idx = date_range('1/1/2000 09:00', periods=6, freq='h', tz=tz, name='idx', unit=unit)
        expected = date_range('1/1/2000 09:00', periods=7, freq='h', tz=tz, name='idx', unit=unit)
        for d in [Timestamp('2000-01-01 15:00', tz=tz), pytz.timezone(tz).localize(datetime(2000, 1, 1, 15))]:
            result = idx.insert(6, d)
            tm.assert_index_equal(result, expected)
            assert result.name == expected.name
            assert result.freq == expected.freq
            assert result.tz == expected.tz
        expected = DatetimeIndex(['2000-01-01 09:00', '2000-01-01 10:00', '2000-01-01 11:00', '2000-01-01 12:00', '2000-01-01 13:00', '2000-01-01 14:00', '2000-01-01 10:00'], name='idx', tz=tz, freq=None).as_unit(unit)
        for d in [Timestamp('2000-01-01 10:00', tz=tz), pytz.timezone(tz).localize(datetime(2000, 1, 1, 10))]:
            result = idx.insert(6, d)
            tm.assert_index_equal(result, expected)
            assert result.name == expected.name
            assert result.tz == expected.tz
            assert result.freq is None