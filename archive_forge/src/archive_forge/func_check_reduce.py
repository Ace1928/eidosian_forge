import numpy as np
import pytest
from pandas.core.dtypes.common import is_bool_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.tests.extension import base
def check_reduce(self, s, op_name, skipna):
    if op_name == 'count':
        result = getattr(s, op_name)()
        expected = getattr(s.astype('float64'), op_name)()
    else:
        result = getattr(s, op_name)(skipna=skipna)
        expected = getattr(s.astype('float64'), op_name)(skipna=skipna)
    if np.isnan(expected):
        expected = pd.NA
    elif op_name in ('min', 'max'):
        expected = bool(expected)
    tm.assert_almost_equal(result, expected)