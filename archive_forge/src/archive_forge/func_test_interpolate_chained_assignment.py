import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('func', ['interpolate', 'ffill', 'bfill'])
def test_interpolate_chained_assignment(using_copy_on_write, func):
    df = DataFrame({'a': [1, np.nan, 2], 'b': 1})
    df_orig = df.copy()
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            getattr(df['a'], func)(inplace=True)
        tm.assert_frame_equal(df, df_orig)
        with tm.raises_chained_assignment_error():
            getattr(df[['a']], func)(inplace=True)
        tm.assert_frame_equal(df, df_orig)
    else:
        with tm.assert_produces_warning(FutureWarning, match='inplace method'):
            getattr(df['a'], func)(inplace=True)
        with tm.assert_produces_warning(None):
            with option_context('mode.chained_assignment', None):
                getattr(df[['a']], func)(inplace=True)
        with tm.assert_produces_warning(None):
            with option_context('mode.chained_assignment', None):
                getattr(df[df['a'] > 1], func)(inplace=True)