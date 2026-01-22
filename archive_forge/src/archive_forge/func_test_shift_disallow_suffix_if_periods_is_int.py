import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_shift_disallow_suffix_if_periods_is_int():
    data = {'a': [1, 2, 3, 4, 5, 6], 'b': [0, 0, 0, 1, 1, 1]}
    df = DataFrame(data)
    msg = 'Cannot specify `suffix` if `periods` is an int.'
    with pytest.raises(ValueError, match=msg):
        df.groupby('b').shift(1, suffix='fails')