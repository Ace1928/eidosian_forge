import numpy as np
import pytest
from pandas._libs.missing import is_matching_na
from pandas.core.dtypes.common import (
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE
def test_frame_from_2d_array(self, data):
    arr2d = data.repeat(2).reshape(-1, 2)
    df = pd.DataFrame(arr2d)
    expected = pd.DataFrame({0: arr2d[:, 0], 1: arr2d[:, 1]})
    tm.assert_frame_equal(df, expected)