from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_type():
    df = DataFrame({'col1': [3, 'string', float], 'col2': [0.25, datetime(2020, 1, 1), np.nan]}, index=['a', 'b', 'c'])
    result = df.apply(type, axis=0)
    expected = Series({'col1': Series, 'col2': Series})
    tm.assert_series_equal(result, expected)
    result = df.apply(type, axis=1)
    expected = Series({'a': Series, 'b': Series, 'c': Series})
    tm.assert_series_equal(result, expected)