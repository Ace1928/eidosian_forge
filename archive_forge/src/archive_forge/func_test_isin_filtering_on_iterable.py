import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import algorithms
from pandas.core.arrays import PeriodArray
@pytest.mark.parametrize('data', [[1, 2, 3], [1.0, 2.0, 3.0]])
@pytest.mark.parametrize('isin', [[1, 2], [1.0, 2.0]])
def test_isin_filtering_on_iterable(data, isin):
    ser = Series(data)
    result = ser.isin((i for i in isin))
    expected_result = Series([True, True, False])
    tm.assert_series_equal(result, expected_result)