import builtins
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_any():
    df = DataFrame([[1, 2, 'foo'], [1, np.nan, 'bar'], [3, np.nan, 'baz']], columns=['A', 'B', 'C'])
    expected = DataFrame([[True, True], [False, True]], columns=['B', 'C'], index=[1, 3])
    expected.index.name = 'A'
    result = df.groupby('A').any()
    tm.assert_frame_equal(result, expected)