import contextlib
import time
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.excel import ExcelWriter
from pandas.io.formats.excel import ExcelFormatter
@pytest.mark.parametrize('engine', ['xlsxwriter', 'openpyxl'])
@pytest.mark.parametrize('css, attrs, expected', shared_style_params)
def test_styler_to_excel_basic_indexes(engine, css, attrs, expected):
    pytest.importorskip(engine)
    df = DataFrame(np.random.default_rng(2).standard_normal((1, 1)))
    styler = df.style
    styler.map_index(lambda x: css, axis=0)
    styler.map_index(lambda x: css, axis=1)
    null_styler = df.style
    null_styler.map(lambda x: 'null: css;')
    null_styler.map_index(lambda x: 'null: css;', axis=0)
    null_styler.map_index(lambda x: 'null: css;', axis=1)
    with tm.ensure_clean('.xlsx') as path:
        with ExcelWriter(path, engine=engine) as writer:
            null_styler.to_excel(writer, sheet_name='null_styled')
            styler.to_excel(writer, sheet_name='styled')
        openpyxl = pytest.importorskip('openpyxl')
        with contextlib.closing(openpyxl.load_workbook(path)) as wb:
            ui_cell, si_cell = (wb['null_styled'].cell(2, 1), wb['styled'].cell(2, 1))
            uc_cell, sc_cell = (wb['null_styled'].cell(1, 2), wb['styled'].cell(1, 2))
        for attr in attrs:
            ui_cell, si_cell = (getattr(ui_cell, attr, None), getattr(si_cell, attr))
            uc_cell, sc_cell = (getattr(uc_cell, attr, None), getattr(sc_cell, attr))
        if isinstance(expected, dict):
            assert ui_cell is None or ui_cell != expected[engine]
            assert si_cell == expected[engine]
            assert uc_cell is None or uc_cell != expected[engine]
            assert sc_cell == expected[engine]
        else:
            assert ui_cell is None or ui_cell != expected
            assert si_cell == expected
            assert uc_cell is None or uc_cell != expected
            assert sc_cell == expected