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
def test_loc_getitem_slicing_datetimes_frame(self):
    df_unique = DataFrame(np.arange(4.0, dtype='float64'), index=[datetime(2001, 1, i, 10, 0) for i in [1, 2, 3, 4]])
    df_dups = DataFrame(np.arange(5.0, dtype='float64'), index=[datetime(2001, 1, i, 10, 0) for i in [1, 2, 2, 3, 4]])
    for df in [df_unique, df_dups]:
        result = df.loc[datetime(2001, 1, 1, 10):]
        tm.assert_frame_equal(result, df)
        result = df.loc[:datetime(2001, 1, 4, 10)]
        tm.assert_frame_equal(result, df)
        result = df.loc[datetime(2001, 1, 1, 10):datetime(2001, 1, 4, 10)]
        tm.assert_frame_equal(result, df)
        result = df.loc[datetime(2001, 1, 1, 11):]
        expected = df.iloc[1:]
        tm.assert_frame_equal(result, expected)
        result = df.loc['20010101 11':]
        tm.assert_frame_equal(result, expected)