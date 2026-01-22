from datetime import datetime
import numpy as np
import pytest
from pandas.errors import MergeError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
def test_join_index_more(float_frame):
    af = float_frame.loc[:, ['A', 'B']]
    bf = float_frame.loc[::2, ['C', 'D']]
    expected = af.copy()
    expected['C'] = float_frame['C'][::2]
    expected['D'] = float_frame['D'][::2]
    result = af.join(bf)
    tm.assert_frame_equal(result, expected)
    result = af.join(bf, how='right')
    tm.assert_frame_equal(result, expected[::2])
    result = bf.join(af, how='right')
    tm.assert_frame_equal(result, expected.loc[:, result.columns])