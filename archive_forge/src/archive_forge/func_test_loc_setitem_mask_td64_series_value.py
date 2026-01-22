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
def test_loc_setitem_mask_td64_series_value(self):
    td1 = Timedelta(0)
    td2 = Timedelta(28767471428571405)
    df = DataFrame({'col': Series([td1, td2])})
    df_copy = df.copy()
    ser = Series([td1])
    expected = df['col'].iloc[1]._value
    df.loc[[True, False]] = ser
    result = df['col'].iloc[1]._value
    assert expected == result
    tm.assert_frame_equal(df, df_copy)