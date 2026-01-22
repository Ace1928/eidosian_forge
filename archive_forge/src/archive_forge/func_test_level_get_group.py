from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_level_get_group(observed):
    df = DataFrame(data=np.arange(2, 22, 2), index=MultiIndex(levels=[CategoricalIndex(['a', 'b']), range(10)], codes=[[0] * 5 + [1] * 5, range(10)], names=['Index1', 'Index2']))
    g = df.groupby(level=['Index1'], observed=observed)
    expected = DataFrame(data=np.arange(2, 12, 2), index=MultiIndex(levels=[CategoricalIndex(['a', 'b']), range(5)], codes=[[0] * 5, range(5)], names=['Index1', 'Index2']))
    msg = 'you will need to pass a length-1 tuple'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = g.get_group('a')
    tm.assert_frame_equal(result, expected)