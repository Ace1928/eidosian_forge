import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_size_period_index():
    ser = Series([1], index=PeriodIndex(['2000'], name='A', freq='D'))
    grp = ser.groupby(level='A')
    result = grp.size()
    tm.assert_series_equal(result, ser)