import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_crosstab_non_aligned(self):
    a = Series([0, 1, 1], index=['a', 'b', 'c'])
    b = Series([3, 4, 3, 4, 3], index=['a', 'b', 'c', 'd', 'f'])
    c = np.array([3, 4, 3], dtype=np.int64)
    expected = DataFrame([[1, 0], [1, 1]], index=Index([0, 1], name='row_0'), columns=Index([3, 4], name='col_0'))
    result = crosstab(a, b)
    tm.assert_frame_equal(result, expected)
    result = crosstab(a, c)
    tm.assert_frame_equal(result, expected)