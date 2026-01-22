import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_with_iterable_basic_functionality(self):
    data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
    shifts = [0, 1, 2]
    df = DataFrame(data)
    shifted = df.shift(shifts)
    expected = DataFrame({'a_0': [1, 2, 3], 'b_0': [4, 5, 6], 'a_1': [np.nan, 1.0, 2.0], 'b_1': [np.nan, 4.0, 5.0], 'a_2': [np.nan, np.nan, 1.0], 'b_2': [np.nan, np.nan, 4.0]})
    tm.assert_frame_equal(expected, shifted)