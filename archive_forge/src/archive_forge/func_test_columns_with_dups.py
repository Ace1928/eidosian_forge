import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_columns_with_dups(self):
    df = DataFrame([[1, 2]], columns=['a', 'a'])
    df.columns = ['a', 'a.1']
    expected = DataFrame([[1, 2]], columns=['a', 'a.1'])
    tm.assert_frame_equal(df, expected)
    df = DataFrame([[1, 2, 3]], columns=['b', 'a', 'a'])
    df.columns = ['b', 'a', 'a.1']
    expected = DataFrame([[1, 2, 3]], columns=['b', 'a', 'a.1'])
    tm.assert_frame_equal(df, expected)