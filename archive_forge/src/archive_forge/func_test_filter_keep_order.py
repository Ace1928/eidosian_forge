import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
def test_filter_keep_order(self):
    df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    result = df.filter(items=['B', 'A'])
    expected = df[['B', 'A']]
    tm.assert_frame_equal(result, expected)