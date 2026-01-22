import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_columns_with_dup_index(self):
    df = DataFrame([[1, 2]], columns=['a', 'a'])
    df.columns = ['b', 'b']
    expected = DataFrame([[1, 2]], columns=['b', 'b'])
    tm.assert_frame_equal(df, expected)