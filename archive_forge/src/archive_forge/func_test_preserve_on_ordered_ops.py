from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('func, values', [('first', ['second', 'first']), ('last', ['fourth', 'third']), ('min', ['fourth', 'first']), ('max', ['second', 'third'])])
def test_preserve_on_ordered_ops(func, values):
    c = Categorical(['first', 'second', 'third', 'fourth'], ordered=True)
    df = DataFrame({'payload': [-1, -2, -1, -2], 'col': c})
    g = df.groupby('payload')
    result = getattr(g, func)()
    expected = DataFrame({'payload': [-2, -1], 'col': Series(values, dtype=c.dtype)}).set_index('payload')
    tm.assert_frame_equal(result, expected)
    sgb = df.groupby('payload')['col']
    result = getattr(sgb, func)()
    expected = expected['col']
    tm.assert_series_equal(result, expected)