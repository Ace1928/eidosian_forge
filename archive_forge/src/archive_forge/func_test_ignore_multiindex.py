import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_ignore_multiindex(self):
    index = pd.MultiIndex.from_tuples([('first', 'second'), ('first', 'third')], names=['baz', 'foobar'])
    df = DataFrame({'foo': [0, 1], 'bar': [2, 3]}, index=index)
    result = melt(df, ignore_index=False)
    expected_index = pd.MultiIndex.from_tuples([('first', 'second'), ('first', 'third')] * 2, names=['baz', 'foobar'])
    expected = DataFrame({'variable': ['foo'] * 2 + ['bar'] * 2, 'value': [0, 1, 2, 3]}, index=expected_index)
    tm.assert_frame_equal(result, expected)