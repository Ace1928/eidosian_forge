import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_transform_frame(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': 1})
    df_orig = df.copy()

    def func(ser):
        ser.iloc[0] = 100
        return ser
    with tm.assert_cow_warning(warn_copy_on_write):
        df.transform(func)
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)