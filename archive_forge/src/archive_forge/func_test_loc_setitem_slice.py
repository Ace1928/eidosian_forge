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
def test_loc_setitem_slice(self):
    df1 = DataFrame({'a': [0, 1, 1], 'b': Series([100, 200, 300], dtype='uint32')})
    ix = df1['a'] == 1
    newb1 = df1.loc[ix, 'b'] + 1
    df1.loc[ix, 'b'] = newb1
    expected = DataFrame({'a': [0, 1, 1], 'b': Series([100, 201, 301], dtype='uint32')})
    tm.assert_frame_equal(df1, expected)
    df2 = DataFrame({'a': [0, 1, 1], 'b': [100, 200, 300]}, dtype='uint64')
    ix = df1['a'] == 1
    newb2 = df2.loc[ix, 'b']
    with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
        df1.loc[ix, 'b'] = newb2
    expected = DataFrame({'a': [0, 1, 1], 'b': [100, 200, 300]}, dtype='uint64')
    tm.assert_frame_equal(df2, expected)