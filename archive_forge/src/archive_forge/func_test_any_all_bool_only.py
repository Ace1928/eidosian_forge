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
def test_any_all_bool_only(self):
    df = DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [None, None, None]}, columns=Index(['col1', 'col2', 'col3'], dtype=object))
    result = df.all(bool_only=True)
    expected = Series(dtype=np.bool_, index=[])
    tm.assert_series_equal(result, expected)
    df = DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [None, None, None], 'col4': [False, False, True]})
    result = df.all(bool_only=True)
    expected = Series({'col4': False})
    tm.assert_series_equal(result, expected)