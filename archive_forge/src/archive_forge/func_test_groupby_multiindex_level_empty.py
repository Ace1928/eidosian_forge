from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_groupby_multiindex_level_empty(self):
    df = DataFrame([[123, 'a', 1.0], [123, 'b', 2.0]], columns=['id', 'category', 'value'])
    df = df.set_index(['id', 'category'])
    empty = df[df.value < 0]
    result = empty.groupby('id').sum()
    expected = DataFrame(dtype='float64', columns=['value'], index=Index([], dtype=np.int64, name='id'))
    tm.assert_frame_equal(result, expected)