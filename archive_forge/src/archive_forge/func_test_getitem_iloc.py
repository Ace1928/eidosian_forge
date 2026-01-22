import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_getitem_iloc(multiindex_dataframe_random_data):
    df = multiindex_dataframe_random_data
    result = df.iloc[2]
    expected = df.xs(df.index[2])
    tm.assert_series_equal(result, expected)