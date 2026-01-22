from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_repr_with_different_nulls_cols(self):
    d = {np.nan: [1, 2], None: [3, 4], NaT: [6, 7], True: [8, 9]}
    df = DataFrame(data=d)
    result = repr(df)
    expected = '   NaN  None  NaT  True\n0    1     3    6     8\n1    2     4    7     9'
    assert result == expected