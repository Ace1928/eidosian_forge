from datetime import (
from functools import partial
from io import BytesIO
import os
import re
import numpy as np
import pytest
from pandas.compat import is_platform_windows
from pandas.compat._constants import PY310
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.excel import (
from pandas.io.excel._util import _writers
def test_write_lists_dict(self, path):
    df = DataFrame({'mixed': ['a', ['b', 'c'], {'d': 'e', 'f': 2}], 'numeric': [1, 2, 3.0], 'str': ['apple', 'banana', 'cherry']})
    df.to_excel(path, sheet_name='Sheet1')
    read = pd.read_excel(path, sheet_name='Sheet1', header=0, index_col=0)
    expected = df.copy()
    expected.mixed = expected.mixed.apply(str)
    expected.numeric = expected.numeric.astype('int64')
    tm.assert_frame_equal(read, expected)