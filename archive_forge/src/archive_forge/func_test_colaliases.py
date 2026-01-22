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
def test_colaliases(self, frame, path):
    frame = frame.copy()
    frame.iloc[:5, frame.columns.get_loc('A')] = np.nan
    frame.to_excel(path, sheet_name='test1')
    frame.to_excel(path, sheet_name='test1', columns=['A', 'B'])
    frame.to_excel(path, sheet_name='test1', header=False)
    frame.to_excel(path, sheet_name='test1', index=False)
    col_aliases = Index(['AA', 'X', 'Y', 'Z'])
    frame.to_excel(path, sheet_name='test1', header=col_aliases)
    with ExcelFile(path) as reader:
        rs = pd.read_excel(reader, sheet_name='test1', index_col=0)
    xp = frame.copy()
    xp.columns = col_aliases
    tm.assert_frame_equal(xp, rs)