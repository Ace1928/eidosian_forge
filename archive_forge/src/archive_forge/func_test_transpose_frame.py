import itertools
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import ExtensionArray
from pandas.core.internals.blocks import EABackedBlock
def test_transpose_frame(self, data):
    df = pd.DataFrame({'A': data[:4], 'B': data[:4]}, index=['a', 'b', 'c', 'd'])
    result = df.T
    expected = pd.DataFrame({'a': type(data)._from_sequence([data[0]] * 2, dtype=data.dtype), 'b': type(data)._from_sequence([data[1]] * 2, dtype=data.dtype), 'c': type(data)._from_sequence([data[2]] * 2, dtype=data.dtype), 'd': type(data)._from_sequence([data[3]] * 2, dtype=data.dtype)}, index=['A', 'B'])
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(np.transpose(np.transpose(df)), df)
    tm.assert_frame_equal(np.transpose(np.transpose(df[['A']])), df[['A']])