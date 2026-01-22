import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_getitem_slice_not_sorted(self, multiindex_dataframe_random_data):
    frame = multiindex_dataframe_random_data
    df = frame.sort_index(level=1).T
    result = df.iloc[:, :np.int32(3)]
    expected = df.reindex(columns=df.columns[:3])
    tm.assert_frame_equal(result, expected)