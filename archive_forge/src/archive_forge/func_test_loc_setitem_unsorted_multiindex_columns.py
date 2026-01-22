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
@pytest.mark.parametrize('key', ['A', ['A'], ('A', slice(None))])
def test_loc_setitem_unsorted_multiindex_columns(self, key):
    mi = MultiIndex.from_tuples([('A', 4), ('B', '3'), ('A', '2')])
    df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=mi)
    obj = df.copy()
    obj.loc[:, key] = np.zeros((2, 2), dtype='int64')
    expected = DataFrame([[0, 2, 0], [0, 5, 0]], columns=mi)
    tm.assert_frame_equal(obj, expected)
    df = df.sort_index(axis=1)
    df.loc[:, key] = np.zeros((2, 2), dtype='int64')
    expected = expected.sort_index(axis=1)
    tm.assert_frame_equal(df, expected)