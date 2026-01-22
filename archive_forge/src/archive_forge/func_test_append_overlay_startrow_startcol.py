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
@pytest.mark.parametrize('startrow, startcol, greeting, goodbye', [(0, 0, ['poop', 'world'], ['goodbye', 'people']), (0, 1, ['hello', 'world'], ['poop', 'people']), (1, 0, ['hello', 'poop'], ['goodbye', 'people']), (1, 1, ['hello', 'world'], ['goodbye', 'poop'])])
def test_append_overlay_startrow_startcol(ext, startrow, startcol, greeting, goodbye):
    df1 = DataFrame({'greeting': ['hello', 'world'], 'goodbye': ['goodbye', 'people']})
    df2 = DataFrame(['poop'])
    with tm.ensure_clean(ext) as f:
        df1.to_excel(f, engine='openpyxl', sheet_name='poo', index=False)
        with ExcelWriter(f, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            df2.to_excel(writer, index=False, header=False, startrow=startrow + 1, startcol=startcol, sheet_name='poo')
        result = pd.read_excel(f, sheet_name='poo', engine='openpyxl')
        expected = DataFrame({'greeting': greeting, 'goodbye': goodbye})
        tm.assert_frame_equal(result, expected)