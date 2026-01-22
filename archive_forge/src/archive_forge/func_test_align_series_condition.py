from datetime import timezone
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_align_series_condition(self):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    result = df[df['a'] == 2]
    expected = DataFrame([[2, 5]], index=[1], columns=['a', 'b'])
    tm.assert_frame_equal(result, expected)
    result = df.where(df['a'] == 2, 0)
    expected = DataFrame({'a': [0, 2, 0], 'b': [0, 5, 0]})
    tm.assert_frame_equal(result, expected)