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
def test_to_string_length(self):
    ser = Series(range(100), dtype='int64')
    res = ser.to_string(max_rows=2, length=True)
    exp = '0      0\n      ..\n99    99\nLength: 100'
    assert res == exp