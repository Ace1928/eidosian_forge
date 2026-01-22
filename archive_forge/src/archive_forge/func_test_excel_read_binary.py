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
def test_excel_read_binary(self, engine, read_ext):
    expected = pd.read_excel('test1' + read_ext, engine=engine)
    with open('test1' + read_ext, 'rb') as f:
        data = f.read()
    actual = pd.read_excel(BytesIO(data), engine=engine)
    tm.assert_frame_equal(expected, actual)