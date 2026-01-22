import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_partial_set_empty_frame_no_index(self):
    expected = DataFrame({0: Series(1, index=range(4))}, columns=['A', 'B', 0])
    df = DataFrame(columns=['A', 'B'])
    df[0] = Series(1, index=range(4))
    tm.assert_frame_equal(df, expected)
    df = DataFrame(columns=['A', 'B'])
    df.loc[:, 0] = Series(1, index=range(4))
    tm.assert_frame_equal(df, expected)