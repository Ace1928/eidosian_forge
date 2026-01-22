import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_groupby_nth_with_column_axis():
    df = DataFrame([[4, 5, 6], [8, 8, 7]], index=['z', 'y'], columns=['C', 'B', 'A'])
    result = df.groupby(df.iloc[1], axis=1).nth(0)
    expected = df.iloc[:, [0, 2]]
    tm.assert_frame_equal(result, expected)