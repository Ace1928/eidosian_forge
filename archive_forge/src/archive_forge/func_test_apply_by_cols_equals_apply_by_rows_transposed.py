from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_by_cols_equals_apply_by_rows_transposed():
    df = DataFrame(np.random.default_rng(2).random([6, 4]), columns=MultiIndex.from_product([['A', 'B'], [1, 2]]))
    msg = "The 'axis' keyword in DataFrame.groupby is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.T.groupby(axis=0, level=0)
    by_rows = gb.apply(lambda x: x.droplevel(axis=0, level=0))
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb2 = df.groupby(axis=1, level=0)
    by_cols = gb2.apply(lambda x: x.droplevel(axis=1, level=0))
    tm.assert_frame_equal(by_cols, by_rows.T)
    tm.assert_frame_equal(by_cols, df)