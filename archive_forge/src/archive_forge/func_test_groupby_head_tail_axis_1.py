import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('op, n, expected_cols', [('head', -1, [0]), ('head', 0, []), ('head', 1, [0, 2]), ('head', 7, [0, 1, 2]), ('tail', -1, [1]), ('tail', 0, []), ('tail', 1, [1, 2]), ('tail', 7, [0, 1, 2])])
def test_groupby_head_tail_axis_1(op, n, expected_cols):
    df = DataFrame([[1, 2, 3], [1, 4, 5], [2, 6, 7], [3, 8, 9]], columns=['A', 'B', 'C'])
    g = df.groupby([0, 0, 1], axis=1)
    expected = df.iloc[:, expected_cols]
    result = getattr(g, op)(n)
    tm.assert_frame_equal(result, expected)