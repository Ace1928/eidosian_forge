import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_partial_set_empty_frame2(self):
    expected = DataFrame(columns=Index(['foo'], dtype=object), index=Index([], dtype='object'))
    df = DataFrame(index=Index([], dtype='object'))
    df['foo'] = Series([], dtype='object')
    tm.assert_frame_equal(df, expected)
    df = DataFrame(index=Index([]))
    df['foo'] = Series(df.index)
    tm.assert_frame_equal(df, expected)
    df = DataFrame(index=Index([]))
    df['foo'] = df.index
    tm.assert_frame_equal(df, expected)