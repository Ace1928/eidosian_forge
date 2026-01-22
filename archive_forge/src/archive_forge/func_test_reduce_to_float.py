import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import FloatingArray
@pytest.mark.parametrize('op', ['mean'])
def test_reduce_to_float(op):
    df = pd.DataFrame({'A': ['a', 'b', 'b'], 'B': [1, None, 3], 'C': pd.array([1, None, 3], dtype='Int64')})
    result = getattr(df.C, op)()
    assert isinstance(result, float)
    result = getattr(df.groupby('A'), op)()
    expected = pd.DataFrame({'B': np.array([1.0, 3.0]), 'C': pd.array([1, 3], dtype='Float64')}, index=pd.Index(['a', 'b'], name='A'))
    tm.assert_frame_equal(result, expected)