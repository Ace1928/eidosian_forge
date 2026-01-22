from copy import deepcopy
import inspect
import pydoc
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._config.config import option_context
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_with_datetimelikes(self):
    df = DataFrame({'A': date_range('20130101', periods=10), 'B': timedelta_range('1 day', periods=10)})
    t = df.T
    result = t.dtypes.value_counts()
    expected = Series({np.dtype('object'): 10}, name='count')
    tm.assert_series_equal(result, expected)