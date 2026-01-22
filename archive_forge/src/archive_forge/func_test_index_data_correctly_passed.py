import numpy as np
import pytest
from pandas.errors import NumbaUtilError
from pandas import (
import pandas._testing as tm
def test_index_data_correctly_passed():
    pytest.importorskip('numba')

    def f(values, index):
        return index - 1
    df = DataFrame({'group': ['A', 'A', 'B'], 'v': [4, 5, 6]}, index=[-1, -2, -3])
    result = df.groupby('group').transform(f, engine='numba')
    expected = DataFrame([-4.0, -3.0, -2.0], columns=['v'], index=[-1, -2, -3])
    tm.assert_frame_equal(result, expected)