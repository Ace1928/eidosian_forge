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
@pytest.mark.parametrize('index', [pd.period_range(start='2017-01-01', end='2018-01-01', freq='M'), timedelta_range(start='1 day', end='2 days', freq='1h')])
def test_loc_getitem_label_slice_period_timedelta(self, index):
    ser = index.to_series()
    result = ser.loc[:index[-2]]
    expected = ser.iloc[:-1]
    tm.assert_series_equal(result, expected)