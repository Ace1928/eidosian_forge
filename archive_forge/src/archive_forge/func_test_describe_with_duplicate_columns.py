import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_describe_with_duplicate_columns(self):
    df = DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=['bar', 'a', 'a'], dtype='float64')
    result = df.describe()
    ser = df.iloc[:, 0].describe()
    expected = pd.concat([ser, ser, ser], keys=df.columns, axis=1)
    tm.assert_frame_equal(result, expected)