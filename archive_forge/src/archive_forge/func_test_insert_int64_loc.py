import numpy as np
import pytest
from pandas.errors import PerformanceWarning
from pandas import (
import pandas._testing as tm
def test_insert_int64_loc(self):
    df = DataFrame({'a': [1, 2]})
    df.insert(np.int64(0), 'b', 0)
    tm.assert_frame_equal(df, DataFrame({'b': [0, 0], 'a': [1, 2]}))