import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
def test_apply_box_td64():
    vals = [pd.Timedelta('1 days'), pd.Timedelta('2 days')]
    ser = Series(vals)
    assert ser.dtype == 'timedelta64[ns]'
    res = ser.apply(lambda x: f'{type(x).__name__}_{x.days}', by_row='compat')
    exp = Series(['Timedelta_1', 'Timedelta_2'])
    tm.assert_series_equal(res, exp)