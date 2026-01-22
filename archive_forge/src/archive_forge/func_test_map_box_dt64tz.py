from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_box_dt64tz(unit):
    vals = [pd.Timestamp('2011-01-01', tz='US/Eastern'), pd.Timestamp('2011-01-02', tz='US/Eastern')]
    ser = Series(vals).dt.as_unit(unit)
    assert ser.dtype == f'datetime64[{unit}, US/Eastern]'
    res = ser.map(lambda x: f'{type(x).__name__}_{x.day}_{x.tz}')
    exp = Series(['Timestamp_1_US/Eastern', 'Timestamp_2_US/Eastern'])
    tm.assert_series_equal(res, exp)