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
def test_to_string_with_col_space(self):
    df = DataFrame(np.random.default_rng(2).random(size=(1, 3)))
    c10 = len(df.to_string(col_space=10).split('\n')[1])
    c20 = len(df.to_string(col_space=20).split('\n')[1])
    c30 = len(df.to_string(col_space=30).split('\n')[1])
    assert c10 < c20 < c30
    with_header = df.to_string(col_space=20)
    with_header_row1 = with_header.splitlines()[1]
    no_header = df.to_string(col_space=20, header=False)
    assert len(with_header_row1) == len(no_header)