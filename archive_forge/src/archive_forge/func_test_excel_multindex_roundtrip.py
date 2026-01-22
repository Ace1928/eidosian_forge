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
@pytest.mark.parametrize('c_idx_names', ['a', None])
@pytest.mark.parametrize('r_idx_names', ['b', None])
@pytest.mark.parametrize('c_idx_levels', [1, 3])
@pytest.mark.parametrize('r_idx_levels', [1, 3])
def test_excel_multindex_roundtrip(self, ext, c_idx_names, r_idx_names, c_idx_levels, r_idx_levels, request):
    with tm.ensure_clean(ext) as pth:
        check_names = bool(r_idx_names) or r_idx_levels <= 1
        if c_idx_levels == 1:
            columns = Index(list('abcde'))
        else:
            columns = MultiIndex.from_arrays([range(5) for _ in range(c_idx_levels)], names=[f'{c_idx_names}-{i}' for i in range(c_idx_levels)])
        if r_idx_levels == 1:
            index = Index(list('ghijk'))
        else:
            index = MultiIndex.from_arrays([range(5) for _ in range(r_idx_levels)], names=[f'{r_idx_names}-{i}' for i in range(r_idx_levels)])
        df = DataFrame(1.1 * np.ones((5, 5)), columns=columns, index=index)
        df.to_excel(pth)
        act = pd.read_excel(pth, index_col=list(range(r_idx_levels)), header=list(range(c_idx_levels)))
        tm.assert_frame_equal(df, act, check_names=check_names)
        df.iloc[0, :] = np.nan
        df.to_excel(pth)
        act = pd.read_excel(pth, index_col=list(range(r_idx_levels)), header=list(range(c_idx_levels)))
        tm.assert_frame_equal(df, act, check_names=check_names)
        df.iloc[-1, :] = np.nan
        df.to_excel(pth)
        act = pd.read_excel(pth, index_col=list(range(r_idx_levels)), header=list(range(c_idx_levels)))
        tm.assert_frame_equal(df, act, check_names=check_names)