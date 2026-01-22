import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_corrwith_dup_cols(self):
    df1 = DataFrame(np.vstack([np.arange(10)] * 3).T)
    df2 = df1.copy()
    df2 = pd.concat((df2, df2[0]), axis=1)
    result = df1.corrwith(df2)
    expected = Series(np.ones(4), index=[0, 0, 1, 2])
    tm.assert_series_equal(result, expected)