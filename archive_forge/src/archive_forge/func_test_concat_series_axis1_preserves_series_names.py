import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_concat_series_axis1_preserves_series_names(self):
    s = Series(np.random.default_rng(2).standard_normal(5), name='A')
    s2 = Series(np.random.default_rng(2).standard_normal(5), name='B')
    result = concat([s, s2], axis=1)
    expected = DataFrame({'A': s, 'B': s2})
    tm.assert_frame_equal(result, expected)
    s2.name = None
    result = concat([s, s2], axis=1)
    tm.assert_index_equal(result.columns, Index(['A', 0], dtype='object'))