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
@pytest.mark.parametrize('np_type', [np.int8, np.int16, np.int32, np.int64])
def test_int_types(self, np_type, path):
    df = DataFrame(np.random.default_rng(2).integers(-10, 10, size=(10, 2)), dtype=np_type)
    df.to_excel(path, sheet_name='test1')
    with ExcelFile(path) as reader:
        recons = pd.read_excel(reader, sheet_name='test1', index_col=0)
    int_frame = df.astype(np.int64)
    tm.assert_frame_equal(int_frame, recons)
    recons2 = pd.read_excel(path, sheet_name='test1', index_col=0)
    tm.assert_frame_equal(int_frame, recons2)