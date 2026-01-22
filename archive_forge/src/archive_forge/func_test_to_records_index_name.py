from collections import abc
import email
from email.parser import Parser
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_records_index_name(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)))
    df.index.name = 'X'
    rs = df.to_records()
    assert 'X' in rs.dtype.fields
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)))
    rs = df.to_records()
    assert 'index' in rs.dtype.fields
    df.index = MultiIndex.from_tuples([('a', 'x'), ('a', 'y'), ('b', 'z')])
    df.index.names = ['A', None]
    result = df.to_records()
    expected = np.rec.fromarrays([np.array(['a', 'a', 'b']), np.array(['x', 'y', 'z'])] + [np.asarray(df.iloc[:, i]) for i in range(3)], dtype={'names': ['A', 'level_1', '0', '1', '2'], 'formats': ['O', 'O', f'{tm.ENDIAN}f8', f'{tm.ENDIAN}f8', f'{tm.ENDIAN}f8']})
    tm.assert_numpy_array_equal(result, expected)