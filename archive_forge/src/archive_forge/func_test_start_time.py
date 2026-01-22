import pytest
from pandas import (
import pandas._testing as tm
def test_start_time(self):
    index = period_range(freq='M', start='2016-01-01', end='2016-05-31')
    expected_index = date_range('2016-01-01', end='2016-05-31', freq='MS')
    tm.assert_index_equal(index.start_time, expected_index)