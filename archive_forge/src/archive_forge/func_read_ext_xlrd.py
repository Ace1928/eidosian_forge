import io
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
import pandas._testing as tm
from pandas.io.excel import ExcelFile
from pandas.io.excel._base import inspect_excel_format
@pytest.fixture(params=['.xls'])
def read_ext_xlrd(request):
    """
    Valid extensions for reading Excel files with xlrd.

    Similar to read_ext, but excludes .ods, .xlsb, and for xlrd>2 .xlsx, .xlsm
    """
    return request.param