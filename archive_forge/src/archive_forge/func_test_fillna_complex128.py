from __future__ import annotations
from datetime import (
import itertools
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('fill_val,fill_dtype', [(1, np.complex128), (1.1, np.complex128), (1 + 1j, np.complex128), (True, object)])
def test_fillna_complex128(self, index_or_series, fill_val, fill_dtype):
    klass = index_or_series
    obj = klass([1 + 1j, np.nan, 3 + 3j, 4 + 4j], dtype=np.complex128)
    assert obj.dtype == np.complex128
    exp = klass([1 + 1j, fill_val, 3 + 3j, 4 + 4j])
    self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)