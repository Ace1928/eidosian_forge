import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
def test_filter_different_dtype(self):
    df = DataFrame({1: [1, 2, 3], 2: [4, 5, 6]})
    result = df.filter(items=['B', 'A'])
    expected = df[[]]
    tm.assert_frame_equal(result, expected)