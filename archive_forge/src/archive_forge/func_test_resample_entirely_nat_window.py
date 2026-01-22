from datetime import datetime
from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
@pytest.mark.parametrize('method, method_args, unit', [('sum', {}, 0), ('sum', {'min_count': 0}, 0), ('sum', {'min_count': 1}, np.nan), ('prod', {}, 1), ('prod', {'min_count': 0}, 1), ('prod', {'min_count': 1}, np.nan)])
def test_resample_entirely_nat_window(method, method_args, unit):
    ser = Series([0] * 2 + [np.nan] * 2, index=date_range('2017', periods=4))
    result = methodcaller(method, **method_args)(ser.resample('2d'))
    exp_dti = pd.DatetimeIndex(['2017-01-01', '2017-01-03'], dtype='M8[ns]', freq='2D')
    expected = Series([0.0, unit], index=exp_dti)
    tm.assert_series_equal(result, expected)