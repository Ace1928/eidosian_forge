from collections import (
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tests.extension.decimal import to_decimal
def test_concat_keys_and_levels(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((1, 3)))
    df2 = DataFrame(np.random.default_rng(2).standard_normal((1, 4)))
    levels = [['foo', 'baz'], ['one', 'two']]
    names = ['first', 'second']
    result = concat([df, df2, df, df2], keys=[('foo', 'one'), ('foo', 'two'), ('baz', 'one'), ('baz', 'two')], levels=levels, names=names)
    expected = concat([df, df2, df, df2])
    exp_index = MultiIndex(levels=levels + [[0]], codes=[[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 0]], names=names + [None])
    expected.index = exp_index
    tm.assert_frame_equal(result, expected)
    result = concat([df, df2, df, df2], keys=[('foo', 'one'), ('foo', 'two'), ('baz', 'one'), ('baz', 'two')], levels=levels)
    assert result.index.names == (None,) * 3
    result = concat([df, df2, df, df2], keys=[('foo', 'one'), ('foo', 'two'), ('baz', 'one'), ('baz', 'two')], names=['first', 'second'])
    assert result.index.names == ('first', 'second', None)
    tm.assert_index_equal(result.index.levels[0], Index(['baz', 'foo'], name='first'))