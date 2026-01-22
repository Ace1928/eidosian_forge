from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
@pytest.mark.parametrize('unit', ['h', 'm', 's', 'ms', 'D', 'M', 'Y'])
def test_frame_setitem_datetime64_col_other_units(self, unit):
    n = 100
    dtype = np.dtype(f'M8[{unit}]')
    vals = np.arange(n, dtype=np.int64).view(dtype)
    if unit in ['s', 'ms']:
        ex_vals = vals
    else:
        ex_vals = vals.astype('datetime64[s]')
    df = DataFrame({'ints': np.arange(n)}, index=np.arange(n))
    df[unit] = vals
    assert df[unit].dtype == ex_vals.dtype
    assert (df[unit].values == ex_vals).all()