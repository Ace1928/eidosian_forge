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
def test_to_excel_periodindex(self, path):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), columns=Index(list('ABCD')), index=date_range('2000-01-01', periods=5, freq='B'))
    xp = df.resample('ME').mean().to_period('M')
    xp.to_excel(path, sheet_name='sht1')
    with ExcelFile(path) as reader:
        rs = pd.read_excel(reader, sheet_name='sht1', index_col=0)
    tm.assert_frame_equal(xp, rs.to_period('M'))