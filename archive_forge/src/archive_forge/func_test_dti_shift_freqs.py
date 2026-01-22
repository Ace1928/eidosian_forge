from datetime import datetime
import pytest
import pytz
from pandas.errors import NullFrequencyError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dti_shift_freqs(self, unit):
    drange = date_range('20130101', periods=5, unit=unit)
    result = drange.shift(1)
    expected = DatetimeIndex(['2013-01-02', '2013-01-03', '2013-01-04', '2013-01-05', '2013-01-06'], dtype=f'M8[{unit}]', freq='D')
    tm.assert_index_equal(result, expected)
    result = drange.shift(-1)
    expected = DatetimeIndex(['2012-12-31', '2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04'], dtype=f'M8[{unit}]', freq='D')
    tm.assert_index_equal(result, expected)
    result = drange.shift(3, freq='2D')
    expected = DatetimeIndex(['2013-01-07', '2013-01-08', '2013-01-09', '2013-01-10', '2013-01-11'], dtype=f'M8[{unit}]', freq='D')
    tm.assert_index_equal(result, expected)