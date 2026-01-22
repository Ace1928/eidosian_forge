from datetime import (
from io import StringIO
import re
import sys
from textwrap import dedent
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data,expected', [({'col1': [1, 2], 'col2': [3, 4]}, '   col1  col2\n0     1     3\n1     2     4'), ({'col1': ['Abc', 0.756], 'col2': [np.nan, 4.5435]}, '    col1    col2\n0    Abc     NaN\n1  0.756  4.5435'), ({'col1': [np.nan, 'a'], 'col2': [0.009, 3.543], 'col3': ['Abc', 23]}, '  col1   col2 col3\n0  NaN  0.009  Abc\n1    a  3.543   23')])
def test_to_string_max_rows_zero(self, data, expected):
    result = DataFrame(data=data).to_string(max_rows=0)
    assert result == expected