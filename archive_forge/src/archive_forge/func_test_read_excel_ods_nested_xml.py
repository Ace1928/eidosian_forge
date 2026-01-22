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
@pytest.mark.parametrize('basename,expected', [('gh-35802', DataFrame({'COLUMN': ['Test (1)']})), ('gh-36122', DataFrame(columns=['got 2nd sa']))])
def test_read_excel_ods_nested_xml(self, engine, read_ext, basename, expected):
    if engine != 'odf':
        pytest.skip(f'Skipped for engine: {engine}')
    actual = pd.read_excel(basename + read_ext)
    tm.assert_frame_equal(actual, expected)