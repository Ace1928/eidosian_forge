import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_subclass_align(self):
    df1 = tm.SubclassedDataFrame({'a': [1, 3, 5], 'b': [1, 3, 5]}, index=list('ACE'))
    df2 = tm.SubclassedDataFrame({'c': [1, 2, 4], 'd': [1, 2, 4]}, index=list('ABD'))
    res1, res2 = df1.align(df2, axis=0)
    exp1 = tm.SubclassedDataFrame({'a': [1, np.nan, 3, np.nan, 5], 'b': [1, np.nan, 3, np.nan, 5]}, index=list('ABCDE'))
    exp2 = tm.SubclassedDataFrame({'c': [1, 2, np.nan, 4, np.nan], 'd': [1, 2, np.nan, 4, np.nan]}, index=list('ABCDE'))
    assert isinstance(res1, tm.SubclassedDataFrame)
    tm.assert_frame_equal(res1, exp1)
    assert isinstance(res2, tm.SubclassedDataFrame)
    tm.assert_frame_equal(res2, exp2)
    res1, res2 = df1.a.align(df2.c)
    assert isinstance(res1, tm.SubclassedSeries)
    tm.assert_series_equal(res1, exp1.a)
    assert isinstance(res2, tm.SubclassedSeries)
    tm.assert_series_equal(res2, exp2.c)