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
def test_loc_getitem_interval_index2(self):
    index = pd.interval_range(start=0, periods=3, closed='both')
    df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=index, columns=['A', 'B', 'C'])
    index_exp = pd.interval_range(start=0, periods=2, freq=1, closed='both')
    expected = Series([1, 4], index=index_exp, name='A')
    result = df.loc[1, 'A']
    tm.assert_series_equal(result, expected)