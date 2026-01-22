from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_different_null_objects(self):
    ser = Series([1, 2, 3, 4], [True, None, np.nan, pd.NaT])
    result = repr(ser)
    expected = 'True    1\nNone    2\nNaN     3\nNaT     4\ndtype: int64'
    assert result == expected