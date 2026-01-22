import io
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
import pandas._testing as tm
from pandas.io.excel import ExcelFile
from pandas.io.excel._base import inspect_excel_format
def test_read_xlsx_fails(datapath):
    from xlrd.biffh import XLRDError
    path = datapath('io', 'data', 'excel', 'test1.xlsx')
    with pytest.raises(XLRDError, match='Excel xlsx file; not supported'):
        pd.read_excel(path, engine='xlrd')