import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_groupby_transform_dtype():
    df = DataFrame({'a': [1], 'val': [1.35]})
    result = df['val'].transform(lambda x: x.map(lambda y: f'+{y}'))
    expected1 = Series(['+1.35'], name='val', dtype='object')
    tm.assert_series_equal(result, expected1)
    result = df.groupby('a')['val'].transform(lambda x: x.map(lambda y: f'+{y}'))
    tm.assert_series_equal(result, expected1)
    result = df.groupby('a')['val'].transform(lambda x: x.map(lambda y: f'+({y})'))
    expected2 = Series(['+(1.35)'], name='val', dtype='object')
    tm.assert_series_equal(result, expected2)
    df['val'] = df['val'].astype(object)
    result = df.groupby('a')['val'].transform(lambda x: x.map(lambda y: f'+{y}'))
    tm.assert_series_equal(result, expected1)