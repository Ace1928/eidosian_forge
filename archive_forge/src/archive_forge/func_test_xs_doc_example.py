import re
import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_xs_doc_example(self):
    arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
    tuples = list(zip(*arrays))
    index = MultiIndex.from_tuples(tuples, names=['first', 'second'])
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 8)), index=['A', 'B', 'C'], columns=index)
    result = df.xs(('one', 'bar'), level=('second', 'first'), axis=1)
    expected = df.iloc[:, [0]]
    tm.assert_frame_equal(result, expected)