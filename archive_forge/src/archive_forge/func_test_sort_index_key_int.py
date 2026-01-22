import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_index_key_int(self):
    df = DataFrame(np.arange(6, dtype='int64'), index=np.arange(6, dtype='int64'))
    result = df.sort_index()
    tm.assert_frame_equal(result, df)
    result = df.sort_index(key=lambda x: -x)
    expected = df.sort_index(ascending=False)
    tm.assert_frame_equal(result, expected)
    result = df.sort_index(key=lambda x: 2 * x)
    tm.assert_frame_equal(result, df)