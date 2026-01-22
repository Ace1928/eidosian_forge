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
@pytest.mark.parametrize('input_array, expected', [('a', 'a'), (['a', 'b'], 'a\nb'), ([1, 'a'], '1\na'), (1, '1'), ([0, -1], ' 0\n-1'), (1.0, '1.0'), ([' a', ' b'], ' a\n b'), (['.1', '1'], '.1\n 1'), (['10', '-10'], ' 10\n-10')])
def test_format_remove_leading_space_series(self, input_array, expected):
    ser = Series(input_array)
    result = ser.to_string(index=False)
    assert result == expected