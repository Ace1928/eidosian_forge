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
def test_to_string_with_column_specific_col_space_raises(self):
    df = DataFrame(np.random.default_rng(2).random(size=(3, 3)), columns=['a', 'b', 'c'])
    msg = 'Col_space length\\(\\d+\\) should match DataFrame number of columns\\(\\d+\\)'
    with pytest.raises(ValueError, match=msg):
        df.to_string(col_space=[30, 40])
    with pytest.raises(ValueError, match=msg):
        df.to_string(col_space=[30, 40, 50, 60])
    msg = 'unknown column'
    with pytest.raises(ValueError, match=msg):
        df.to_string(col_space={'a': 'foo', 'b': 23, 'd': 34})