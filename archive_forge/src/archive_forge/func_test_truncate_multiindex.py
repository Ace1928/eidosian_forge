import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_truncate_multiindex(self, frame_or_series):
    mi = pd.MultiIndex.from_product([[1, 2, 3, 4], ['A', 'B']], names=['L1', 'L2'])
    s1 = DataFrame(range(mi.shape[0]), index=mi, columns=['col'])
    s1 = tm.get_obj(s1, frame_or_series)
    result = s1.truncate(before=2, after=3)
    df = DataFrame.from_dict({'L1': [2, 2, 3, 3], 'L2': ['A', 'B', 'A', 'B'], 'col': [2, 3, 4, 5]})
    expected = df.set_index(['L1', 'L2'])
    expected = tm.get_obj(expected, frame_or_series)
    tm.assert_equal(result, expected)