import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_axis_one_empty(self):
    df = DataFrame()
    result = df.shift(1, axis=1)
    tm.assert_frame_equal(result, df)