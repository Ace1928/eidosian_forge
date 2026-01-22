from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_loc_setitem_ndframe_values_alignment(self, using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df.loc[[False, False, True], ['a']] = DataFrame({'a': [10, 20, 30]}, index=[2, 1, 0])
    expected = DataFrame({'a': [1, 2, 10], 'b': [4, 5, 6]})
    tm.assert_frame_equal(df, expected)
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df.loc[[False, False, True], ['a']] = Series([10, 11, 12], index=[2, 1, 0])
    tm.assert_frame_equal(df, expected)
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df.loc[[False, False, True], 'a'] = Series([10, 11, 12], index=[2, 1, 0])
    tm.assert_frame_equal(df, expected)
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df_orig = df.copy()
    ser = df['a']
    with tm.assert_cow_warning(warn_copy_on_write):
        ser.loc[[False, False, True]] = Series([10, 11, 12], index=[2, 1, 0])
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)
    else:
        tm.assert_frame_equal(df, expected)