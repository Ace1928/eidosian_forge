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
def test_read_excel_skiprows_callable_not_in(self, request, engine, read_ext):
    xfail_datetimes_with_pyxlsb(engine, request)
    unit = get_exp_unit(read_ext, engine)
    actual = pd.read_excel('testskiprows' + read_ext, sheet_name='skiprows_list', skiprows=lambda x: x not in [1, 3, 5])
    expected = DataFrame([[1, 2.5, pd.Timestamp('2015-01-01'), True], [3, 4.5, pd.Timestamp('2015-01-03'), False]], columns=['a', 'b', 'c', 'd'])
    expected['c'] = expected['c'].astype(f'M8[{unit}]')
    tm.assert_frame_equal(actual, expected)