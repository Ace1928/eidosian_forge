import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
def test_loc_multiindex_columns_one_level(self):
    df = DataFrame([[1, 2]], columns=[['a', 'b']])
    expected = DataFrame([1], columns=[['a']])
    result = df['a']
    tm.assert_frame_equal(result, expected)
    result = df.loc[:, 'a']
    tm.assert_frame_equal(result, expected)