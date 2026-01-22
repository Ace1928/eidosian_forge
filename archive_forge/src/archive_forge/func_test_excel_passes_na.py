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
def test_excel_passes_na(self, read_ext):
    with pd.ExcelFile('test4' + read_ext) as excel:
        parsed = pd.read_excel(excel, sheet_name='Sheet1', keep_default_na=False, na_values=['apple'])
    expected = DataFrame([['NA'], [1], ['NA'], [np.nan], ['rabbit']], columns=['Test'])
    tm.assert_frame_equal(parsed, expected)
    with pd.ExcelFile('test4' + read_ext) as excel:
        parsed = pd.read_excel(excel, sheet_name='Sheet1', keep_default_na=True, na_values=['apple'])
    expected = DataFrame([[np.nan], [1], [np.nan], [np.nan], ['rabbit']], columns=['Test'])
    tm.assert_frame_equal(parsed, expected)
    with pd.ExcelFile('test5' + read_ext) as excel:
        parsed = pd.read_excel(excel, sheet_name='Sheet1', keep_default_na=False, na_values=['apple'])
    expected = DataFrame([['1.#QNAN'], [1], ['nan'], [np.nan], ['rabbit']], columns=['Test'])
    tm.assert_frame_equal(parsed, expected)
    with pd.ExcelFile('test5' + read_ext) as excel:
        parsed = pd.read_excel(excel, sheet_name='Sheet1', keep_default_na=True, na_values=['apple'])
    expected = DataFrame([[np.nan], [1], [np.nan], [np.nan], ['rabbit']], columns=['Test'])
    tm.assert_frame_equal(parsed, expected)