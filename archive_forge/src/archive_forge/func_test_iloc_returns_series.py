import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('indexer, expected', [(lambda df: df.iloc[0], lambda arr: Series(arr[0], index=[[2, 2, 4], [6, 8, 10]], name=(4, 8))), (lambda df: df.iloc[2], lambda arr: Series(arr[2], index=[[2, 2, 4], [6, 8, 10]], name=(8, 12))), (lambda df: df.iloc[:, 2], lambda arr: Series(arr[:, 2], index=[[4, 4, 8], [8, 10, 12]], name=(4, 10)))])
def test_iloc_returns_series(indexer, expected, simple_multiindex_dataframe):
    df = simple_multiindex_dataframe
    arr = df.values
    result = indexer(df)
    expected = expected(arr)
    tm.assert_series_equal(result, expected)