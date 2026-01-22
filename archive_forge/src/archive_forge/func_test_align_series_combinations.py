from datetime import timezone
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_align_series_combinations(self):
    df = DataFrame({'a': [1, 3, 5], 'b': [1, 3, 5]}, index=list('ACE'))
    s = Series([1, 2, 4], index=list('ABD'), name='x')
    res1, res2 = df.align(s, axis=0)
    exp1 = DataFrame({'a': [1, np.nan, 3, np.nan, 5], 'b': [1, np.nan, 3, np.nan, 5]}, index=list('ABCDE'))
    exp2 = Series([1, 2, np.nan, 4, np.nan], index=list('ABCDE'), name='x')
    tm.assert_frame_equal(res1, exp1)
    tm.assert_series_equal(res2, exp2)
    res1, res2 = s.align(df)
    tm.assert_series_equal(res1, exp2)
    tm.assert_frame_equal(res2, exp1)