import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setattr_columns_vs_construct_with_columns(self):
    arr = np.random.default_rng(2).standard_normal((3, 2))
    idx = list(range(2))
    df = DataFrame(arr, columns=['A', 'A'])
    df.columns = idx
    expected = DataFrame(arr, columns=idx)
    tm.assert_frame_equal(df, expected)