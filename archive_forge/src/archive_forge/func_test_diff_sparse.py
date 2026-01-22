import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_diff_sparse(self):
    sparse_df = DataFrame([[0, 1], [1, 0]], dtype='Sparse[int]')
    result = sparse_df.diff()
    expected = DataFrame([[np.nan, np.nan], [1.0, -1.0]], dtype=pd.SparseDtype('float', 0.0))
    tm.assert_frame_equal(result, expected)