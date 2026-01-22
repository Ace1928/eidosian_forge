import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import algorithms
from pandas.core.arrays import PeriodArray
@pytest.mark.parametrize('data,is_in', [([1, [2]], [1]), (['simple str', [{'values': 3}]], ['simple str'])])
def test_isin_filtering_with_mixed_object_types(data, is_in):
    ser = Series(data)
    result = ser.isin(is_in)
    expected = Series([True, False])
    tm.assert_series_equal(result, expected)