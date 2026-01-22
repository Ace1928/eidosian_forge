import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_index_multiindex_sparse_column(self):
    expected = DataFrame({i: pd.array([0.0, 0.0, 0.0, 0.0], dtype=pd.SparseDtype('float64', 0.0)) for i in range(4)}, index=MultiIndex.from_product([[1, 2], [1, 2]]))
    result = expected.sort_index(level=0)
    tm.assert_frame_equal(result, expected)