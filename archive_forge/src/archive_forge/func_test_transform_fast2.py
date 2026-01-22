import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_fast2():
    df = DataFrame({'grouping': [0, 1, 1, 3], 'f': [1.1, 2.1, 3.1, 4.5], 'd': date_range('2014-1-1', '2014-1-4'), 'i': [1, 2, 3, 4]}, columns=['grouping', 'f', 'i', 'd'])
    result = df.groupby('grouping').transform('first')
    dates = Index([Timestamp('2014-1-1'), Timestamp('2014-1-2'), Timestamp('2014-1-2'), Timestamp('2014-1-4')], dtype='M8[ns]')
    expected = DataFrame({'f': [1.1, 2.1, 2.1, 4.5], 'd': dates, 'i': [1, 2, 2, 4]}, columns=['f', 'i', 'd'])
    tm.assert_frame_equal(result, expected)
    result = df.groupby('grouping')[['f', 'i']].transform('first')
    expected = expected[['f', 'i']]
    tm.assert_frame_equal(result, expected)