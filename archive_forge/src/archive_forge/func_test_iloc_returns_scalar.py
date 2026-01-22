import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_iloc_returns_scalar(simple_multiindex_dataframe):
    df = simple_multiindex_dataframe
    arr = df.values
    result = df.iloc[2, 2]
    expected = arr[2, 2]
    assert result == expected