import numpy as np
import pytest
from pandas.errors import PerformanceWarning
from pandas import (
import pandas._testing as tm
def test_insert_column_bug_4032(self):
    df = DataFrame({'b': [1.1, 2.2]})
    df = df.rename(columns={})
    df.insert(0, 'a', [1, 2])
    result = df.rename(columns={})
    expected = DataFrame([[1, 1.1], [2, 2.2]], columns=['a', 'b'])
    tm.assert_frame_equal(result, expected)
    df.insert(0, 'c', [1.3, 2.3])
    result = df.rename(columns={})
    expected = DataFrame([[1.3, 1, 1.1], [2.3, 2, 2.2]], columns=['c', 'a', 'b'])
    tm.assert_frame_equal(result, expected)