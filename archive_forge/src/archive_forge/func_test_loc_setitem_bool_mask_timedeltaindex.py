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
def test_loc_setitem_bool_mask_timedeltaindex(self):
    df = DataFrame({'x': range(10)})
    df.index = to_timedelta(range(10), unit='s')
    conditions = [df['x'] > 3, df['x'] == 3, df['x'] < 3]
    expected_data = [[0, 1, 2, 3, 10, 10, 10, 10, 10, 10], [0, 1, 2, 10, 4, 5, 6, 7, 8, 9], [10, 10, 10, 3, 4, 5, 6, 7, 8, 9]]
    for cond, data in zip(conditions, expected_data):
        result = df.copy()
        result.loc[cond, 'x'] = 10
        expected = DataFrame(data, index=to_timedelta(range(10), unit='s'), columns=['x'], dtype='int64')
        tm.assert_frame_equal(expected, result)