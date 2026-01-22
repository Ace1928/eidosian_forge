import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_values_multicolumn_uint64(self):
    df = DataFrame({'a': pd.Series([18446637057563306014, 1162265347240853609]), 'b': pd.Series([1, 2])})
    df['a'] = df['a'].astype(np.uint64)
    result = df.sort_values(['a', 'b'])
    expected = DataFrame({'a': pd.Series([18446637057563306014, 1162265347240853609]), 'b': pd.Series([1, 2])}, index=pd.Index([1, 0]))
    tm.assert_frame_equal(result, expected)