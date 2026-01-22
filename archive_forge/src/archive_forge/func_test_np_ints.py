import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_np_ints(slice_test_df, slice_test_grouped):
    result = slice_test_grouped.nth(np.array([0, 1]))
    expected = slice_test_df.iloc[[0, 1, 2, 3, 4]]
    tm.assert_frame_equal(result, expected)