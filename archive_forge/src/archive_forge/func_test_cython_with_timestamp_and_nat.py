import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.mark.parametrize('op', ['first', 'last', 'max', 'min'])
@pytest.mark.parametrize('data', [Timestamp('2016-10-14 21:00:44.557'), Timedelta('17088 days 21:00:44.557')])
def test_cython_with_timestamp_and_nat(op, data):
    df = DataFrame({'a': [0, 1], 'b': [data, NaT]})
    index = Index([0, 1], name='a')
    expected = DataFrame({'b': [data, NaT]}, index=index)
    result = df.groupby('a').aggregate(op)
    tm.assert_frame_equal(expected, result)