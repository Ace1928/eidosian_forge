import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
def test_describe_strs(self):
    ser = Series(['a', 'a', 'b', 'c', 'd'], name='str_data')
    result = ser.describe()
    expected = Series([5, 4, 'a', 2], name='str_data', index=['count', 'unique', 'top', 'freq'])
    tm.assert_series_equal(result, expected)