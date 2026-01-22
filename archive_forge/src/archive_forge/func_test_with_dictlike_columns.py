from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_with_dictlike_columns():
    df = DataFrame([[1, 2], [1, 2]], columns=['a', 'b'])
    result = df.apply(lambda x: {'s': x['a'] + x['b']}, axis=1)
    expected = Series([{'s': 3} for t in df.itertuples()])
    tm.assert_series_equal(result, expected)
    df['tm'] = [Timestamp('2017-05-01 00:00:00'), Timestamp('2017-05-02 00:00:00')]
    result = df.apply(lambda x: {'s': x['a'] + x['b']}, axis=1)
    tm.assert_series_equal(result, expected)
    result = (df['a'] + df['b']).apply(lambda x: {'s': x})
    expected = Series([{'s': 3}, {'s': 3}])
    tm.assert_series_equal(result, expected)