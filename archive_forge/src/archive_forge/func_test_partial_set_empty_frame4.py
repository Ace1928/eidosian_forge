import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_partial_set_empty_frame4(self):
    df = DataFrame(index=Index([], dtype='int64'))
    df['foo'] = range(len(df))
    expected = DataFrame(columns=Index(['foo'], dtype=object), index=Index([], dtype='int64'))
    expected['foo'] = expected['foo'].astype('int64')
    tm.assert_frame_equal(df, expected)