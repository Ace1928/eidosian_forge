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
def test_loc_setitem_cast2(self):
    df = DataFrame(np.random.default_rng(2).random((30, 3)), columns=tuple('ABC'))
    df['event'] = np.nan
    with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
        df.loc[10, 'event'] = 'foo'
    result = df.dtypes
    expected = Series([np.dtype('float64')] * 3 + [np.dtype('object')], index=['A', 'B', 'C', 'event'])
    tm.assert_series_equal(result, expected)