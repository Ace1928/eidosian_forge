import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
@pytest.mark.parametrize('copy', [False, True])
def test_dict_nocopy(self, request, copy, any_numeric_ea_dtype, any_numpy_dtype, using_array_manager, using_copy_on_write):
    if using_array_manager and (not copy) and (any_numpy_dtype not in tm.STRING_DTYPES + tm.BYTES_DTYPES):
        td.mark_array_manager_not_yet_implemented(request)
    a = np.array([1, 2], dtype=any_numpy_dtype)
    b = np.array([3, 4], dtype=any_numpy_dtype)
    if b.dtype.kind in ['S', 'U']:
        pytest.skip(f'{b.dtype} get cast, making the checks below more cumbersome')
    c = pd.array([1, 2], dtype=any_numeric_ea_dtype)
    c_orig = c.copy()
    df = DataFrame({'a': a, 'b': b, 'c': c}, copy=copy)

    def get_base(obj):
        if isinstance(obj, np.ndarray):
            return obj.base
        elif isinstance(obj.dtype, np.dtype):
            return obj._ndarray.base
        else:
            raise TypeError

    def check_views(c_only: bool=False):
        assert sum((x is c for x in df._mgr.arrays)) == 1
        if c_only:
            return
        assert sum((get_base(x) is a for x in df._mgr.arrays if isinstance(x.dtype, np.dtype))) == 1
        assert sum((get_base(x) is b for x in df._mgr.arrays if isinstance(x.dtype, np.dtype))) == 1
    if not copy:
        check_views()
    if lib.is_np_dtype(df.dtypes.iloc[0], 'fciuO'):
        warn = None
    else:
        warn = FutureWarning
    with tm.assert_produces_warning(warn, match='incompatible dtype'):
        df.iloc[0, 0] = 0
        df.iloc[0, 1] = 0
    if not copy:
        check_views(True)
    df.iloc[:, 2] = pd.array([45, 46], dtype=c.dtype)
    assert df.dtypes.iloc[2] == c.dtype
    if not copy and (not using_copy_on_write):
        check_views(True)
    if copy:
        if a.dtype.kind == 'M':
            assert a[0] == a.dtype.type(1, 'ns')
            assert b[0] == b.dtype.type(3, 'ns')
        else:
            assert a[0] == a.dtype.type(1)
            assert b[0] == b.dtype.type(3)
        assert c[0] == c_orig[0]
    elif not using_copy_on_write:
        assert c[0] == 45