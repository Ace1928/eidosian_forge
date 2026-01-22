import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_subsets_multiindex_dtype(self):
    data = [['x', 1]]
    columns = [('a', 'b', np.nan), ('a', 'c', 0.0)]
    df = DataFrame(data, columns=MultiIndex.from_tuples(columns))
    expected = df.dtypes.a.b
    result = df.a.b.dtypes
    tm.assert_series_equal(result, expected)