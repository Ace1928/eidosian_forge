from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
def test_insert2(self, unit):
    idx = date_range('1/1/2000', periods=3, freq='ME', name='idx', unit=unit)
    expected_0 = DatetimeIndex(['1999-12-31', '2000-01-31', '2000-02-29', '2000-03-31'], name='idx', freq='ME').as_unit(unit)
    expected_3 = DatetimeIndex(['2000-01-31', '2000-02-29', '2000-03-31', '2000-04-30'], name='idx', freq='ME').as_unit(unit)
    expected_1_nofreq = DatetimeIndex(['2000-01-31', '2000-01-31', '2000-02-29', '2000-03-31'], name='idx', freq=None).as_unit(unit)
    expected_3_nofreq = DatetimeIndex(['2000-01-31', '2000-02-29', '2000-03-31', '2000-01-02'], name='idx', freq=None).as_unit(unit)
    cases = [(0, datetime(1999, 12, 31), expected_0), (-3, datetime(1999, 12, 31), expected_0), (3, datetime(2000, 4, 30), expected_3), (1, datetime(2000, 1, 31), expected_1_nofreq), (3, datetime(2000, 1, 2), expected_3_nofreq)]
    for n, d, expected in cases:
        result = idx.insert(n, d)
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name
        assert result.freq == expected.freq