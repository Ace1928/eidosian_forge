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
@pytest.mark.parametrize('index,box,expected', [(([0, 2], ['A', 'B', 'C', 'D']), 7, DataFrame([[7, 7, 7, 7], [3, 4, np.nan, np.nan], [7, 7, 7, 7]], columns=['A', 'B', 'C', 'D'])), ((1, ['C', 'D']), [7, 8], DataFrame([[1, 2, np.nan, np.nan], [3, 4, 7, 8], [5, 6, np.nan, np.nan]], columns=['A', 'B', 'C', 'D'])), ((1, ['A', 'B', 'C']), np.array([7, 8, 9], dtype=np.int64), DataFrame([[1, 2, np.nan], [7, 8, 9], [5, 6, np.nan]], columns=['A', 'B', 'C'])), ((slice(1, 3, None), ['B', 'C', 'D']), [[7, 8, 9], [10, 11, 12]], DataFrame([[1, 2, np.nan, np.nan], [3, 7, 8, 9], [5, 10, 11, 12]], columns=['A', 'B', 'C', 'D'])), ((slice(1, 3, None), ['C', 'A', 'D']), np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int64), DataFrame([[1, 2, np.nan, np.nan], [8, 4, 7, 9], [11, 6, 10, 12]], columns=['A', 'B', 'C', 'D'])), ((slice(None, None, None), ['A', 'C']), DataFrame([[7, 8], [9, 10], [11, 12]], columns=['A', 'C']), DataFrame([[7, 2, 8], [9, 4, 10], [11, 6, 12]], columns=['A', 'B', 'C']))])
def test_loc_setitem_missing_columns(self, index, box, expected):
    df = DataFrame([[1, 2], [3, 4], [5, 6]], columns=['A', 'B'])
    df.loc[index] = box
    tm.assert_frame_equal(df, expected)