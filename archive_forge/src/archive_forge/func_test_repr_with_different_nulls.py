from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_repr_with_different_nulls(self):
    df = DataFrame([1, 2, 3, 4], [True, None, np.nan, NaT])
    result = repr(df)
    expected = '      0\nTrue  1\nNone  2\nNaN   3\nNaT   4'
    assert result == expected