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
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason="sum doesn't work for arrow strings")
def test_reduce_mixed_frame(self):
    df = DataFrame({'bool_data': [True, True, False, False, False], 'int_data': [10, 20, 30, 40, 50], 'string_data': ['a', 'b', 'c', 'd', 'e']})
    df.reindex(columns=['bool_data', 'int_data', 'string_data'])
    test = df.sum(axis=0)
    tm.assert_numpy_array_equal(test.values, np.array([2, 150, 'abcde'], dtype=object))
    alt = df.T.sum(axis=1)
    tm.assert_series_equal(test, alt)