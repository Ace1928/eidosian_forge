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
def test_to_string_header_false(self):
    df = DataFrame([1, 2])
    df.index.name = 'a'
    s = df.to_string(header=False)
    expected = 'a   \n0  1\n1  2'
    assert s == expected
    df = DataFrame([[1, 2], [3, 4]])
    df.index.name = 'a'
    s = df.to_string(header=False)
    expected = 'a      \n0  1  2\n1  3  4'
    assert s == expected