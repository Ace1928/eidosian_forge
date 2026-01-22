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
@pytest.mark.parametrize('mode,expected', [('w', ['baz']), ('a', ['foo', 'bar', 'baz'])])
def test_write_append_mode(ext, mode, expected):
    df = DataFrame([1], columns=['baz'])
    with tm.ensure_clean(ext) as f:
        wb = openpyxl.Workbook()
        wb.worksheets[0].title = 'foo'
        wb.worksheets[0]['A1'].value = 'foo'
        wb.create_sheet('bar')
        wb.worksheets[1]['A1'].value = 'bar'
        wb.save(f)
        with ExcelWriter(f, engine='openpyxl', mode=mode) as writer:
            df.to_excel(writer, sheet_name='baz', index=False)
        with contextlib.closing(openpyxl.load_workbook(f)) as wb2:
            result = [sheet.title for sheet in wb2.worksheets]
            assert result == expected
            for index, cell_value in enumerate(expected):
                assert wb2.worksheets[index]['A1'].value == cell_value