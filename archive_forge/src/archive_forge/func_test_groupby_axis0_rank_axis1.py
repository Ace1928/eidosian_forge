from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_groupby_axis0_rank_axis1():
    df = DataFrame({0: [1, 3, 5, 7], 1: [2, 4, 6, 8], 2: [1.5, 3.5, 5.5, 7.5]}, index=['a', 'a', 'b', 'b'])
    gb = df.groupby(level=0, axis=0)
    res = gb.rank(axis=1)
    expected = concat([df.loc['a'].rank(axis=1), df.loc['b'].rank(axis=1)], axis=0)
    tm.assert_frame_equal(res, expected)
    alt = gb.rank(axis=0)
    assert not alt.equals(expected)