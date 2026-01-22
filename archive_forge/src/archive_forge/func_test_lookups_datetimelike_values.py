import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('vals', [pd.date_range('2016-01-01', periods=3), pd.timedelta_range('1 Day', periods=3)])
def test_lookups_datetimelike_values(self, vals, dtype):
    ser = Series(vals, index=range(3, 6))
    ser.index = ser.index.astype(dtype)
    expected = vals[1]
    result = ser[4.0]
    assert isinstance(result, type(expected)) and result == expected
    result = ser[4]
    assert isinstance(result, type(expected)) and result == expected
    result = ser.loc[4.0]
    assert isinstance(result, type(expected)) and result == expected
    result = ser.loc[4]
    assert isinstance(result, type(expected)) and result == expected
    result = ser.at[4.0]
    assert isinstance(result, type(expected)) and result == expected
    result = ser.at[4]
    assert isinstance(result, type(expected)) and result == expected
    result = ser.iloc[1]
    assert isinstance(result, type(expected)) and result == expected
    result = ser.iat[1]
    assert isinstance(result, type(expected)) and result == expected