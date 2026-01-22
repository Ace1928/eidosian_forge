import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_20476():
    mi = MultiIndex.from_product([['A', 'B'], ['a', 'b', 'c']])
    df = DataFrame(-1, index=range(3), columns=mi)
    filler = DataFrame([[1, 2, 3.0]] * 3, index=range(3), columns=['a', 'b', 'c'])
    df['A'] = filler
    expected = DataFrame({0: [1, 1, 1], 1: [2, 2, 2], 2: [3.0, 3.0, 3.0], 3: [-1, -1, -1], 4: [-1, -1, -1], 5: [-1, -1, -1]})
    expected.columns = mi
    exp_dtypes = Series([np.dtype(np.int64)] * 2 + [np.dtype(np.float64)] + [np.dtype(np.int64)] * 3, index=mi)
    tm.assert_series_equal(df.dtypes, exp_dtypes)