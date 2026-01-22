import pytest
from pandas import (
import pandas._testing as tm
def test_end_time(self):
    index = period_range(freq='M', start='2016-01-01', end='2016-05-31')
    expected_index = date_range('2016-01-01', end='2016-05-31', freq='ME')
    expected_index += Timedelta(1, 'D') - Timedelta(1, 'ns')
    tm.assert_index_equal(index.end_time, expected_index)