import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_float(self, float_frame):
    casted = float_frame.astype(int)
    expected = DataFrame(float_frame.values.astype(int), index=float_frame.index, columns=float_frame.columns)
    tm.assert_frame_equal(casted, expected)
    casted = float_frame.astype(np.int32)
    expected = DataFrame(float_frame.values.astype(np.int32), index=float_frame.index, columns=float_frame.columns)
    tm.assert_frame_equal(casted, expected)
    float_frame['foo'] = '5'
    casted = float_frame.astype(int)
    expected = DataFrame(float_frame.values.astype(int), index=float_frame.index, columns=float_frame.columns)
    tm.assert_frame_equal(casted, expected)