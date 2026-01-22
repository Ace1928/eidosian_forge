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
def test_read_excel_multiindex(self, request, engine, read_ext):
    xfail_datetimes_with_pyxlsb(engine, request)
    unit = get_exp_unit(read_ext, engine)
    mi = MultiIndex.from_product([['foo', 'bar'], ['a', 'b']])
    mi_file = 'testmultiindex' + read_ext
    expected = DataFrame([[1, 2.5, pd.Timestamp('2015-01-01'), True], [2, 3.5, pd.Timestamp('2015-01-02'), False], [3, 4.5, pd.Timestamp('2015-01-03'), False], [4, 5.5, pd.Timestamp('2015-01-04'), True]], columns=mi)
    expected[mi[2]] = expected[mi[2]].astype(f'M8[{unit}]')
    actual = pd.read_excel(mi_file, sheet_name='mi_column', header=[0, 1], index_col=0)
    tm.assert_frame_equal(actual, expected)
    expected.index = mi
    expected.columns = ['a', 'b', 'c', 'd']
    actual = pd.read_excel(mi_file, sheet_name='mi_index', index_col=[0, 1])
    tm.assert_frame_equal(actual, expected)
    expected.columns = mi
    actual = pd.read_excel(mi_file, sheet_name='both', index_col=[0, 1], header=[0, 1])
    tm.assert_frame_equal(actual, expected)
    expected.columns = ['a', 'b', 'c', 'd']
    expected.index = mi.set_names(['ilvl1', 'ilvl2'])
    actual = pd.read_excel(mi_file, sheet_name='mi_index_name', index_col=[0, 1])
    tm.assert_frame_equal(actual, expected)
    expected.index = list(range(4))
    expected.columns = mi.set_names(['c1', 'c2'])
    actual = pd.read_excel(mi_file, sheet_name='mi_column_name', header=[0, 1], index_col=0)
    tm.assert_frame_equal(actual, expected)
    expected.columns = mi.set_levels([1, 2], level=1).set_names(['c1', 'c2'])
    actual = pd.read_excel(mi_file, sheet_name='name_with_int', index_col=0, header=[0, 1])
    tm.assert_frame_equal(actual, expected)
    expected.columns = mi.set_names(['c1', 'c2'])
    expected.index = mi.set_names(['ilvl1', 'ilvl2'])
    actual = pd.read_excel(mi_file, sheet_name='both_name', index_col=[0, 1], header=[0, 1])
    tm.assert_frame_equal(actual, expected)
    actual = pd.read_excel(mi_file, sheet_name='both_name_skiprows', index_col=[0, 1], header=[0, 1], skiprows=2)
    tm.assert_frame_equal(actual, expected)