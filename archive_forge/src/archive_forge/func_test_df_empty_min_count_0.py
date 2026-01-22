from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
@pytest.mark.parametrize('opname, dtype, exp_value, exp_dtype', [('sum', np.int8, 0, np.int64), ('prod', np.int8, 1, np.int_), ('sum', np.int64, 0, np.int64), ('prod', np.int64, 1, np.int64), ('sum', np.uint8, 0, np.uint64), ('prod', np.uint8, 1, np.uint), ('sum', np.uint64, 0, np.uint64), ('prod', np.uint64, 1, np.uint64), ('sum', np.float32, 0, np.float32), ('prod', np.float32, 1, np.float32), ('sum', np.float64, 0, np.float64)])
def test_df_empty_min_count_0(self, opname, dtype, exp_value, exp_dtype):
    df = DataFrame({0: [], 1: []}, dtype=dtype)
    result = getattr(df, opname)(min_count=0)
    expected = Series([exp_value, exp_value], dtype=exp_dtype)
    tm.assert_series_equal(result, expected)