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
@pytest.mark.parametrize('fill_val,exp_dtype', [(Timestamp('2022-01-06'), 'datetime64[ns]'), (Timestamp('2022-01-07', tz='US/Eastern'), 'datetime64[ns, US/Eastern]')])
def test_loc_setitem_using_datetimelike_str_as_index(fill_val, exp_dtype):
    data = ['2022-01-02', '2022-01-03', '2022-01-04', fill_val.date()]
    index = DatetimeIndex(data, tz=fill_val.tz, dtype=exp_dtype)
    df = DataFrame([10, 11, 12, 14], columns=['a'], index=index)
    df.loc['2022-01-08', 'a'] = 13
    data.append('2022-01-08')
    expected_index = DatetimeIndex(data, dtype=exp_dtype)
    tm.assert_index_equal(df.index, expected_index, exact=True)