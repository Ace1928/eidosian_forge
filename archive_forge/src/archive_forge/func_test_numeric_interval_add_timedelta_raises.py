from datetime import timedelta
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('interval', [Interval(1, 2), Interval(1.0, 2.0)])
@pytest.mark.parametrize('delta', [Timedelta(days=7), timedelta(7), np.timedelta64(7, 'D')])
def test_numeric_interval_add_timedelta_raises(self, interval, delta):
    msg = '|'.join(['unsupported operand', 'cannot use operands', 'Only numeric, Timestamp and Timedelta endpoints are allowed'])
    with pytest.raises((TypeError, ValueError), match=msg):
        interval + delta
    with pytest.raises((TypeError, ValueError), match=msg):
        delta + interval