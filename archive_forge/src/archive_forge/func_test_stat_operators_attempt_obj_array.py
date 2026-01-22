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
@pytest.mark.parametrize('method', ['sum', 'mean', 'prod', 'var', 'std', 'skew', 'min', 'max'])
@pytest.mark.parametrize('df', [DataFrame({'a': [-0.0004998754019959134, -0.001646725777291983, 0.0006769587077588301], 'b': [-0, -0, 0.0], 'c': [0.00031111847529610595, 0.0014902627951905339, -0.0009409920003597969]}, index=['foo', 'bar', 'baz'], dtype='O'), DataFrame({0: [np.nan, 2], 1: [np.nan, 3], 2: [np.nan, 4]}, dtype=object)])
@pytest.mark.filterwarnings('ignore:Mismatched null-like values:FutureWarning')
def test_stat_operators_attempt_obj_array(self, method, df, axis):
    assert df.values.dtype == np.object_
    result = getattr(df, method)(axis=axis)
    expected = getattr(df.astype('f8'), method)(axis=axis).astype(object)
    if axis in [1, 'columns'] and method in ['min', 'max']:
        expected[expected.isna()] = None
    tm.assert_series_equal(result, expected)