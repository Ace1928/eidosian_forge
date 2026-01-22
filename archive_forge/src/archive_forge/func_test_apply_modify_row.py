import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_apply_modify_row(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'A': [1, 2], 'B': [3, 4]})
    df_orig = df.copy()

    def transform(row):
        row['B'] = 100
        return row
    with tm.assert_cow_warning(warn_copy_on_write):
        df.apply(transform, axis=1)
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)
    else:
        assert df.loc[0, 'B'] == 100
    df = DataFrame({'A': [1, 2], 'B': ['b', 'c']})
    df_orig = df.copy()
    with tm.assert_produces_warning(None):
        df.apply(transform, axis=1)
    tm.assert_frame_equal(df, df_orig)