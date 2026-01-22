from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_drop_duplicates_with_duplicate_column_names():
    df = DataFrame([[1, 2, 5], [3, 4, 6], [3, 4, 7]], columns=['a', 'a', 'b'])
    result0 = df.drop_duplicates()
    tm.assert_frame_equal(result0, df)
    result1 = df.drop_duplicates('a')
    expected1 = df[:2]
    tm.assert_frame_equal(result1, expected1)