import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_setitem_with_expansion_non_category(self, df):
    df.loc['a'] = 20
    df3 = df.copy()
    df3.loc['d', 'A'] = 10
    bidx3 = Index(list('aabbcad'), name='B')
    expected3 = DataFrame({'A': [20, 20, 2, 3, 4, 20, 10.0]}, index=Index(bidx3))
    tm.assert_frame_equal(df3, expected3)
    df4 = df.copy()
    df4.loc['d', 'C'] = 10
    expected3 = DataFrame({'A': [20, 20, 2, 3, 4, 20, np.nan], 'C': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10]}, index=Index(bidx3))
    tm.assert_frame_equal(df4, expected3)