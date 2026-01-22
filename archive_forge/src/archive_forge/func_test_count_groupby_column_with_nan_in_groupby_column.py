from itertools import product
from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_count_groupby_column_with_nan_in_groupby_column(self):
    df = DataFrame({'A': [1, 1, 1, 1, 1], 'B': [5, 4, np.nan, 3, 0]})
    res = df.groupby(['B']).count()
    expected = DataFrame(index=Index([0.0, 3.0, 4.0, 5.0], name='B'), data={'A': [1, 1, 1, 1]})
    tm.assert_frame_equal(expected, res)