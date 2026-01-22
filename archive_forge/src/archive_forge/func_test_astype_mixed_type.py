import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_mixed_type(self):
    df = DataFrame({'a': 1.0, 'b': 2, 'c': 'foo', 'float32': np.array([1.0] * 10, dtype='float32'), 'int32': np.array([1] * 10, dtype='int32')}, index=np.arange(10))
    mn = df._get_numeric_data().copy()
    mn['little_float'] = np.array(12345.0, dtype='float16')
    mn['big_float'] = np.array(123456789101112.0, dtype='float64')
    casted = mn.astype('float64')
    _check_cast(casted, 'float64')
    casted = mn.astype('int64')
    _check_cast(casted, 'int64')
    casted = mn.reindex(columns=['little_float']).astype('float16')
    _check_cast(casted, 'float16')
    casted = mn.astype('float32')
    _check_cast(casted, 'float32')
    casted = mn.astype('int32')
    _check_cast(casted, 'int32')
    casted = mn.astype('O')
    _check_cast(casted, 'object')