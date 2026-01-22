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
@pytest.mark.parametrize('max_cols, max_rows, expected', [(10, None, ' 0   1   2   3   4   ...  6   7   8   9   10\n  0   0   0   0   0  ...   0   0   0   0   0\n  0   0   0   0   0  ...   0   0   0   0   0\n  0   0   0   0   0  ...   0   0   0   0   0\n  0   0   0   0   0  ...   0   0   0   0   0'), (None, 2, ' 0   1   2   3   4   5   6   7   8   9   10\n  0   0   0   0   0   0   0   0   0   0   0\n ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..\n  0   0   0   0   0   0   0   0   0   0   0'), (10, 2, ' 0   1   2   3   4   ...  6   7   8   9   10\n  0   0   0   0   0  ...   0   0   0   0   0\n ..  ..  ..  ..  ..  ...  ..  ..  ..  ..  ..\n  0   0   0   0   0  ...   0   0   0   0   0'), (9, 2, ' 0   1   2   3   ...  7   8   9   10\n  0   0   0   0  ...   0   0   0   0\n ..  ..  ..  ..  ...  ..  ..  ..  ..\n  0   0   0   0  ...   0   0   0   0'), (1, 1, ' 0  ...\n 0  ...\n..  ...')])
def test_truncation_no_index(self, max_cols, max_rows, expected):
    df = DataFrame([[0] * 11] * 4)
    assert df.to_string(index=False, max_cols=max_cols, max_rows=max_rows) == expected