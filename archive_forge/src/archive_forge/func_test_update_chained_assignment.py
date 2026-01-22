import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_update_chained_assignment(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3]})
    ser2 = Series([100.0], index=[1])
    df_orig = df.copy()
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            df['a'].update(ser2)
        tm.assert_frame_equal(df, df_orig)
        with tm.raises_chained_assignment_error():
            df[['a']].update(ser2.to_frame())
        tm.assert_frame_equal(df, df_orig)
    else:
        with tm.assert_produces_warning(FutureWarning, match='inplace method'):
            df['a'].update(ser2)
        with tm.assert_produces_warning(None):
            with option_context('mode.chained_assignment', None):
                df[['a']].update(ser2.to_frame())
        with tm.assert_produces_warning(None):
            with option_context('mode.chained_assignment', None):
                df[df['a'] > 1].update(ser2.to_frame())