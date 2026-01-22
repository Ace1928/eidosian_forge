import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
def test_fillna_dict_inplace_nonunique_columns(self, using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'A': [np.nan] * 3, 'B': [NaT, Timestamp(1), NaT], 'C': [np.nan, 'foo', 2]})
    df.columns = ['A', 'A', 'A']
    orig = df[:]
    with tm.assert_cow_warning(warn_copy_on_write):
        df.fillna({'A': 2}, inplace=True)
    expected = DataFrame({'A': [2.0] * 3, 'B': [2, Timestamp(1), 2], 'C': [2, 'foo', 2]})
    expected.columns = ['A', 'A', 'A']
    tm.assert_frame_equal(df, expected)
    if not using_copy_on_write:
        assert tm.shares_memory(df.iloc[:, 0], orig.iloc[:, 0])
    assert not tm.shares_memory(df.iloc[:, 1], orig.iloc[:, 1])
    if not using_copy_on_write:
        assert tm.shares_memory(df.iloc[:, 2], orig.iloc[:, 2])