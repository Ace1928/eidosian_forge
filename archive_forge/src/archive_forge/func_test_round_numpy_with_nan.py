import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_round_numpy_with_nan(self):
    df = Series([1.53, np.nan, 0.06]).to_frame()
    with tm.assert_produces_warning(None):
        result = df.round()
    expected = Series([2.0, np.nan, 0.0]).to_frame()
    tm.assert_frame_equal(result, expected)