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
def test_loc_setitem_time_key(self, using_array_manager):
    index = date_range('2012-01-01', '2012-01-05', freq='30min')
    df = DataFrame(np.random.default_rng(2).standard_normal((len(index), 5)), index=index)
    akey = time(12, 0, 0)
    bkey = slice(time(13, 0, 0), time(14, 0, 0))
    ainds = [24, 72, 120, 168]
    binds = [26, 27, 28, 74, 75, 76, 122, 123, 124, 170, 171, 172]
    result = df.copy()
    result.loc[akey] = 0
    result = result.loc[akey]
    expected = df.loc[akey].copy()
    expected.loc[:] = 0
    if using_array_manager:
        expected = expected.astype(float)
    tm.assert_frame_equal(result, expected)
    result = df.copy()
    result.loc[akey] = 0
    result.loc[akey] = df.iloc[ainds]
    tm.assert_frame_equal(result, df)
    result = df.copy()
    result.loc[bkey] = 0
    result = result.loc[bkey]
    expected = df.loc[bkey].copy()
    expected.loc[:] = 0
    if using_array_manager:
        expected = expected.astype(float)
    tm.assert_frame_equal(result, expected)
    result = df.copy()
    result.loc[bkey] = 0
    result.loc[bkey] = df.iloc[binds]
    tm.assert_frame_equal(result, df)