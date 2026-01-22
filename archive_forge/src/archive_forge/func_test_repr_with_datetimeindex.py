from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_repr_with_datetimeindex(self):
    df = DataFrame({'A': [1, 2, 3]}, index=date_range('2000', periods=3))
    result = repr(df)
    expected = '            A\n2000-01-01  1\n2000-01-02  2\n2000-01-03  3'
    assert result == expected