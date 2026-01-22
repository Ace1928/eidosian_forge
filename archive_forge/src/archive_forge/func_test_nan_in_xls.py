import io
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
import pandas._testing as tm
from pandas.io.excel import ExcelFile
from pandas.io.excel._base import inspect_excel_format
def test_nan_in_xls(datapath):
    path = datapath('io', 'data', 'excel', 'test6.xls')
    expected = pd.DataFrame({0: np.r_[0, 2].astype('int64'), 1: np.r_[1, np.nan]})
    result = pd.read_excel(path, header=None)
    tm.assert_frame_equal(result, expected)