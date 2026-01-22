import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_diff_axis1_nonconsolidated(self):
    df = DataFrame({'y': Series([2]), 'z': Series([3])})
    df.insert(0, 'x', 1)
    result = df.diff(axis=1)
    expected = DataFrame({'x': np.nan, 'y': Series(1), 'z': Series(1)})
    tm.assert_frame_equal(result, expected)