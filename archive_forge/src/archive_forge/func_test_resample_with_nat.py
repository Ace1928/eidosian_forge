from datetime import timedelta
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.timedeltas import timedelta_range
def test_resample_with_nat():
    index = pd.to_timedelta(['0s', pd.NaT, '2s'])
    result = DataFrame({'value': [2, 3, 5]}, index).resample('1s').mean()
    expected = DataFrame({'value': [2.5, np.nan, 5.0]}, index=timedelta_range('0 day', periods=3, freq='1s'))
    tm.assert_frame_equal(result, expected)