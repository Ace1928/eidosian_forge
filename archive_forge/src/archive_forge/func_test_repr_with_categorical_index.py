from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_repr_with_categorical_index(self):
    df = DataFrame({'A': [1, 2, 3]}, index=CategoricalIndex(['a', 'b', 'c']))
    result = repr(df)
    expected = '   A\na  1\nb  2\nc  3'
    assert result == expected