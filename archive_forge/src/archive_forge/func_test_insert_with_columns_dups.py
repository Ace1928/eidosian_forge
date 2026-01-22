import numpy as np
import pytest
from pandas.errors import PerformanceWarning
from pandas import (
import pandas._testing as tm
def test_insert_with_columns_dups(self):
    df = DataFrame()
    df.insert(0, 'A', ['g', 'h', 'i'], allow_duplicates=True)
    df.insert(0, 'A', ['d', 'e', 'f'], allow_duplicates=True)
    df.insert(0, 'A', ['a', 'b', 'c'], allow_duplicates=True)
    exp = DataFrame([['a', 'd', 'g'], ['b', 'e', 'h'], ['c', 'f', 'i']], columns=['A', 'A', 'A'])
    tm.assert_frame_equal(df, exp)