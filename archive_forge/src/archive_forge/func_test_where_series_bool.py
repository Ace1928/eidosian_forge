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
@pytest.mark.parametrize('fill_val,exp_dtype', [(1, object), (1.1, object), (1 + 1j, object), (True, np.bool_)])
def test_where_series_bool(self, index_or_series, fill_val, exp_dtype):
    klass = index_or_series
    obj = klass([True, False, True, False])
    assert obj.dtype == np.bool_
    self._run_test(obj, fill_val, klass, exp_dtype)