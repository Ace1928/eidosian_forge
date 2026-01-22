import contextlib
from pathlib import Path
import re
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.excel import (
from pandas.io.excel._openpyxl import OpenpyxlReader
@pytest.mark.parametrize('data_only, expected', [(True, 0), (False, '=1+1')])
def test_engine_kwargs_append_data_only(ext, data_only, expected):
    with tm.ensure_clean(ext) as f:
        DataFrame(['=1+1']).to_excel(f)
        with ExcelWriter(f, engine='openpyxl', mode='a', engine_kwargs={'data_only': data_only}) as writer:
            assert writer.sheets['Sheet1']['B2'].value == expected
            DataFrame().to_excel(writer, sheet_name='Sheet2')
        assert pd.read_excel(f, sheet_name='Sheet1', engine='openpyxl', engine_kwargs={'data_only': data_only}).iloc[0, 1] == expected