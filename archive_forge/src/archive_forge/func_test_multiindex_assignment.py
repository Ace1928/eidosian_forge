import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_multiindex_assignment(self):
    df = DataFrame(np.random.default_rng(2).integers(5, 10, size=9).reshape(3, 3), columns=list('abc'), index=[[4, 4, 8], [8, 10, 12]])
    df['d'] = np.nan
    arr = np.array([0.0, 1.0])
    df.loc[4, 'd'] = arr
    tm.assert_series_equal(df.loc[4, 'd'], Series(arr, index=[8, 10], name='d'))