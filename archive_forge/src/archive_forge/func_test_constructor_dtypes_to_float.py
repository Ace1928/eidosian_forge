from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('vals', [[1, 2, 3], [1.0, 2.0, 3.0], np.array([1.0, 2.0, 3.0]), np.array([1, 2, 3], dtype=int), np.array([1.0, 2.0, 3.0], dtype=float)])
def test_constructor_dtypes_to_float(self, vals, float_numpy_dtype):
    dtype = float_numpy_dtype
    index = Index(vals, dtype=dtype)
    assert index.dtype == dtype