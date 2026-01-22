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
def test_to_string_small_float_values(self):
    df = DataFrame({'a': [1.5, 1e-17, -5.5e-07]})
    result = df.to_string()
    if _three_digit_exp():
        expected = '               a\n0  1.500000e+000\n1  1.000000e-017\n2 -5.500000e-007'
    else:
        expected = '              a\n0  1.500000e+00\n1  1.000000e-17\n2 -5.500000e-07'
    assert result == expected
    df = df * 0
    result = df.to_string()
    expected = '   0\n0  0\n1  0\n2 -0'