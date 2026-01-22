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
def test_sheets(self, frame, path):
    unit = get_exp_unit(path)
    tsframe = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), columns=Index(list('ABCD')), index=date_range('2000-01-01', periods=5, freq='B'))
    index = pd.DatetimeIndex(np.asarray(tsframe.index), freq=None)
    tsframe.index = index
    expected = tsframe[:]
    expected.index = expected.index.as_unit(unit)
    frame = frame.copy()
    frame.iloc[:5, frame.columns.get_loc('A')] = np.nan
    frame.to_excel(path, sheet_name='test1')
    frame.to_excel(path, sheet_name='test1', columns=['A', 'B'])
    frame.to_excel(path, sheet_name='test1', header=False)
    frame.to_excel(path, sheet_name='test1', index=False)
    with ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name='test1')
        tsframe.to_excel(writer, sheet_name='test2')
    with ExcelFile(path) as reader:
        recons = pd.read_excel(reader, sheet_name='test1', index_col=0)
        tm.assert_frame_equal(frame, recons)
        recons = pd.read_excel(reader, sheet_name='test2', index_col=0)
    tm.assert_frame_equal(expected, recons)
    assert 2 == len(reader.sheet_names)
    assert 'test1' == reader.sheet_names[0]
    assert 'test2' == reader.sheet_names[1]