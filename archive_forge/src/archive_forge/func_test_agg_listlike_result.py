from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_agg_listlike_result():
    df = DataFrame({'A': [2, 2, 3], 'B': [1.5, np.nan, 1.5], 'C': ['foo', None, 'bar']})

    def func(group_col):
        return list(group_col.dropna().unique())
    result = df.agg(func)
    expected = Series([[2, 3], [1.5], ['foo', 'bar']], index=['A', 'B', 'C'])
    tm.assert_series_equal(result, expected)
    result = df.agg([func])
    expected = expected.to_frame('func').T
    tm.assert_frame_equal(result, expected)