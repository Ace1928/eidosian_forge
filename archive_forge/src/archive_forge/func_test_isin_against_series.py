import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_isin_against_series(self):
    df = DataFrame({'A': [1, 2, 3, 4], 'B': [2, np.nan, 4, 4]}, index=['a', 'b', 'c', 'd'])
    s = Series([1, 3, 11, 4], index=['a', 'b', 'c', 'd'])
    expected = DataFrame(False, index=df.index, columns=df.columns)
    expected.loc['a', 'A'] = True
    expected.loc['d'] = True
    result = df.isin(s)
    tm.assert_frame_equal(result, expected)