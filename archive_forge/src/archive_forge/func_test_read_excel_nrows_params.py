from __future__ import annotations
from datetime import (
from functools import partial
from io import BytesIO
import os
from pathlib import Path
import platform
import re
from urllib.error import URLError
from zipfile import BadZipFile
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('filename,sheet_name,header,index_col,skiprows', [('testmultiindex', 'mi_column', [0, 1], 0, None), ('testmultiindex', 'mi_index', None, [0, 1], None), ('testmultiindex', 'both', [0, 1], [0, 1], None), ('testmultiindex', 'mi_column_name', [0, 1], 0, None), ('testskiprows', 'skiprows_list', None, None, [0, 2]), ('testskiprows', 'skiprows_list', None, None, lambda x: x in (0, 2))])
def test_read_excel_nrows_params(self, read_ext, filename, sheet_name, header, index_col, skiprows):
    """
        For various parameters, we should get the same result whether we
        limit the rows during load (nrows=3) or after (df.iloc[:3]).
        """
    expected = pd.read_excel(filename + read_ext, sheet_name=sheet_name, header=header, index_col=index_col, skiprows=skiprows).iloc[:3]
    actual = pd.read_excel(filename + read_ext, sheet_name=sheet_name, header=header, index_col=index_col, skiprows=skiprows, nrows=3)
    tm.assert_frame_equal(actual, expected)