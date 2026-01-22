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
@pytest.mark.parametrize('usecols', [[0, 1, 3], [0, 3, 1], [1, 0, 3], [1, 3, 0], [3, 0, 1], [3, 1, 0]])
def test_usecols_diff_positional_int_columns_order(self, request, engine, read_ext, usecols, df_ref):
    xfail_datetimes_with_pyxlsb(engine, request)
    expected = df_ref[['A', 'C']]
    adjust_expected(expected, read_ext, engine)
    result = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, usecols=usecols)
    tm.assert_frame_equal(result, expected)