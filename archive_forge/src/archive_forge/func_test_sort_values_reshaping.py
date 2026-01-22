import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_values_reshaping(self):
    values = list(range(21))
    expected = DataFrame([values], columns=values)
    df = expected.sort_values(expected.index[0], axis=1, ignore_index=True)
    tm.assert_frame_equal(df, expected)