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
def test_to_string_line_width_with_both_index_and_header(self):
    df = DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    df_s = df.to_string(line_width=1)
    expected = '   x  \\\n0  1   \n1  2   \n2  3   \n\n   y  \n0  4  \n1  5  \n2  6  '
    assert df_s == expected
    df = DataFrame({'x': [11, 22, 33], 'y': [4, 5, 6]})
    df_s = df.to_string(line_width=1)
    expected = '    x  \\\n0  11   \n1  22   \n2  33   \n\n   y  \n0  4  \n1  5  \n2  6  '
    assert df_s == expected
    df = DataFrame({'x': [11, 22, -33], 'y': [4, 5, -6]})
    df_s = df.to_string(line_width=1)
    expected = '    x  \\\n0  11   \n1  22   \n2 -33   \n\n   y  \n0  4  \n1  5  \n2 -6  '
    assert df_s == expected