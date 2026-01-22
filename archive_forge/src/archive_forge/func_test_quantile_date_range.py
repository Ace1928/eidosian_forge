import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_quantile_date_range(self, interp_method, request, using_array_manager):
    interpolation, method = interp_method
    dti = pd.date_range('2016-01-01', periods=3, tz='US/Pacific')
    ser = Series(dti)
    df = DataFrame(ser)
    result = df.quantile(numeric_only=False, interpolation=interpolation, method=method)
    expected = Series(['2016-01-02 00:00:00'], name=0.5, dtype='datetime64[ns, US/Pacific]')
    if method == 'table' and using_array_manager:
        request.applymarker(pytest.mark.xfail(reason='Axis name incorrectly set.'))
    tm.assert_series_equal(result, expected)