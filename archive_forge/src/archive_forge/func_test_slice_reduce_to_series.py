from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_slice_reduce_to_series(self):
    df = DataFrame({'A': range(24)}, index=date_range('2000', periods=24, freq='ME'))
    expected = Series(range(12), index=date_range('2000', periods=12, freq='ME'), name='A')
    result = df.loc['2000', 'A']
    tm.assert_series_equal(result, expected)