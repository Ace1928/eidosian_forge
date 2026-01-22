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
def test_read_excel_parse_dates(self, ext):
    df = DataFrame({'col': [1, 2, 3], 'date_strings': date_range('2012-01-01', periods=3)})
    df2 = df.copy()
    df2['date_strings'] = df2['date_strings'].dt.strftime('%m/%d/%Y')
    with tm.ensure_clean(ext) as pth:
        df2.to_excel(pth)
        res = pd.read_excel(pth, index_col=0)
        tm.assert_frame_equal(df2, res)
        res = pd.read_excel(pth, parse_dates=['date_strings'], index_col=0)
        tm.assert_frame_equal(df, res)
        date_parser = lambda x: datetime.strptime(x, '%m/%d/%Y')
        with tm.assert_produces_warning(FutureWarning, match="use 'date_format' instead", raise_on_extra_warnings=False):
            res = pd.read_excel(pth, parse_dates=['date_strings'], date_parser=date_parser, index_col=0)
        tm.assert_frame_equal(df, res)
        res = pd.read_excel(pth, parse_dates=['date_strings'], date_format='%m/%d/%Y', index_col=0)
        tm.assert_frame_equal(df, res)