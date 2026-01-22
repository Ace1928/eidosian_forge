from itertools import product
from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('datetimelike', [[Timestamp(f'2016-05-{i:02d} 20:09:25+00:00') for i in range(1, 4)], [Timestamp(f'2016-05-{i:02d} 20:09:25') for i in range(1, 4)], [Timestamp(f'2016-05-{i:02d} 20:09:25', tz='UTC') for i in range(1, 4)], [Timedelta(x, unit='h') for x in range(1, 4)], [Period(freq='2W', year=2017, month=x) for x in range(1, 4)]])
def test_count_with_datetimelike(self, datetimelike):
    df = DataFrame({'x': ['a', 'a', 'b'], 'y': datetimelike})
    res = df.groupby('x').count()
    expected = DataFrame({'y': [2, 1]}, index=['a', 'b'])
    expected.index.name = 'x'
    tm.assert_frame_equal(expected, res)