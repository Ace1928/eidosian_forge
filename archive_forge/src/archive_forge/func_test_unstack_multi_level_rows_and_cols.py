from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
def test_unstack_multi_level_rows_and_cols(self):
    df = DataFrame([[1, 2], [3, 4], [-1, -2], [-3, -4]], columns=MultiIndex.from_tuples([['a', 'b', 'c'], ['d', 'e', 'f']]), index=MultiIndex.from_tuples([['m1', 'P3', 222], ['m1', 'A5', 111], ['m2', 'P3', 222], ['m2', 'A5', 111]], names=['i1', 'i2', 'i3']))
    result = df.unstack(['i3', 'i2'])
    expected = df.unstack(['i3']).unstack(['i2'])
    tm.assert_frame_equal(result, expected)