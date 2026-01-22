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
@pytest.mark.parametrize('fill_val, fill_dtype', [(1, object), (1.1, object), (1 + 1j, object), (True, object)])
def test_fillna_object(self, index_or_series, fill_val, fill_dtype):
    klass = index_or_series
    obj = klass(['a', np.nan, 'c', 'd'], dtype=object)
    assert obj.dtype == object
    exp = klass(['a', fill_val, 'c', 'd'], dtype=object)
    self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)