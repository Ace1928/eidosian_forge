import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
def test_getitem_boolean_frame_with_duplicate_columns(self, df_dup_cols):
    df = DataFrame(np.arange(12).reshape(3, 4), columns=['A', 'B', 'C', 'D'], dtype='float64')
    expected = df[df > 6]
    expected.columns = df_dup_cols.columns
    df = df_dup_cols
    result = df[df > 6]
    tm.assert_frame_equal(result, expected)