import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_corrwith_spearman_with_tied_data(self):
    pytest.importorskip('scipy')
    df1 = DataFrame({'A': [1, np.nan, 7, 8], 'B': [False, True, True, False], 'C': [10, 4, 9, 3]})
    df2 = df1[['B', 'C']]
    result = (df1 + 1).corrwith(df2.B, method='spearman')
    expected = Series([0.0, 1.0, 0.0], index=['A', 'B', 'C'])
    tm.assert_series_equal(result, expected)
    df_bool = DataFrame({'A': [True, True, False, False], 'B': [True, False, False, True]})
    ser_bool = Series([True, True, False, True])
    result = df_bool.corrwith(ser_bool)
    expected = Series([0.57735, 0.57735], index=['A', 'B'])
    tm.assert_series_equal(result, expected)