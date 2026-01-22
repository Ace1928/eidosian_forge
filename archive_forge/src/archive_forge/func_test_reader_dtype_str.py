from __future__ import annotations
from datetime import (
from functools import partial
from io import BytesIO
import os
from pathlib import Path
import platform
import re
from urllib.error import URLError
from zipfile import BadZipFile
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('dtype,expected', [(None, DataFrame({'a': [1, 2, 3, 4], 'b': [2.5, 3.5, 4.5, 5.5], 'c': [1, 2, 3, 4], 'd': [1.0, 2.0, np.nan, 4.0]})), ({'a': 'float64', 'b': 'float32', 'c': str, 'd': str}, DataFrame({'a': Series([1, 2, 3, 4], dtype='float64'), 'b': Series([2.5, 3.5, 4.5, 5.5], dtype='float32'), 'c': Series(['001', '002', '003', '004'], dtype=object), 'd': Series(['1', '2', np.nan, '4'], dtype=object)}))])
def test_reader_dtype_str(self, read_ext, dtype, expected):
    basename = 'testdtype'
    actual = pd.read_excel(basename + read_ext, dtype=dtype)
    tm.assert_frame_equal(actual, expected)