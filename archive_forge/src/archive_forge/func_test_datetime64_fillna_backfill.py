from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_datetime64_fillna_backfill(self):
    ser = Series([NaT, NaT, '2013-08-05 15:30:00.000001'], dtype='M8[ns]')
    expected = Series(['2013-08-05 15:30:00.000001', '2013-08-05 15:30:00.000001', '2013-08-05 15:30:00.000001'], dtype='M8[ns]')
    result = ser.fillna(method='backfill')
    tm.assert_series_equal(result, expected)