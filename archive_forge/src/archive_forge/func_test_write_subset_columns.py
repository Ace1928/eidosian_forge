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
@pytest.mark.parametrize('to_excel_index,read_excel_index_col', [(True, 0), (False, None)])
def test_write_subset_columns(self, path, to_excel_index, read_excel_index_col):
    write_frame = DataFrame({'A': [1, 1, 1], 'B': [2, 2, 2], 'C': [3, 3, 3]})
    write_frame.to_excel(path, sheet_name='col_subset_bug', columns=['A', 'B'], index=to_excel_index)
    expected = write_frame[['A', 'B']]
    read_frame = pd.read_excel(path, sheet_name='col_subset_bug', index_col=read_excel_index_col)
    tm.assert_frame_equal(expected, read_frame)