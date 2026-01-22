import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_with_iterable_check_other_arguments(self):
    data = {'a': [1, 2], 'b': [4, 5]}
    shifts = [0, 1]
    df = DataFrame(data)
    shifted = df[['a']].shift(shifts, suffix='_suffix')
    expected = DataFrame({'a_suffix_0': [1, 2], 'a_suffix_1': [np.nan, 1.0]})
    tm.assert_frame_equal(shifted, expected)
    msg = 'If `periods` contains multiple shifts, `axis` cannot be 1.'
    with pytest.raises(ValueError, match=msg):
        df.shift(shifts, axis=1)
    msg = "Periods must be integer, but s is <class 'str'>."
    with pytest.raises(TypeError, match=msg):
        df.shift(['s'])
    msg = 'If `periods` is an iterable, it cannot be empty.'
    with pytest.raises(ValueError, match=msg):
        df.shift([])
    msg = 'Cannot specify `suffix` if `periods` is an int.'
    with pytest.raises(ValueError, match=msg):
        df.shift(1, suffix='fails')