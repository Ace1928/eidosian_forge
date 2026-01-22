import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
def test_getitem_slice_same_dim_only_one_axis(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 8)))
    result = df.iloc[slice(None, None, 2),]
    assert result.shape == (5, 8)
    expected = df.iloc[slice(None, None, 2), slice(None)]
    tm.assert_frame_equal(result, expected)