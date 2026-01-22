import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_ffill():
    data = [['a', 0.0], ['a', float('nan')], ['b', 1.0], ['b', float('nan')]]
    df = DataFrame(data, columns=['key', 'values'])
    result = df.groupby('key').transform('ffill')
    expected = DataFrame({'values': [0.0, 0.0, 1.0, 1.0]})
    tm.assert_frame_equal(result, expected)
    result = df.groupby('key')['values'].transform('ffill')
    expected = Series([0.0, 0.0, 1.0, 1.0], name='values')
    tm.assert_series_equal(result, expected)