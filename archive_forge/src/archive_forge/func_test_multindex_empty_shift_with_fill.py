import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_multindex_empty_shift_with_fill():
    df = DataFrame(columns=['a', 'b', 'c'])
    shifted = df.groupby(['a', 'b']).shift(1)
    shifted_with_fill = df.groupby(['a', 'b']).shift(1, fill_value=0)
    tm.assert_frame_equal(shifted, shifted_with_fill)
    tm.assert_index_equal(shifted.index, shifted_with_fill.index)