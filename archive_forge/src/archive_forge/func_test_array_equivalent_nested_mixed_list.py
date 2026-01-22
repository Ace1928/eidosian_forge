from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas._config import config as cf
from pandas._libs import missing as libmissing
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore:elementwise comparison failed:DeprecationWarning')
@pytest.mark.xfail(reason='failing')
@pytest.mark.parametrize('strict_nan', [True, False])
def test_array_equivalent_nested_mixed_list(strict_nan):
    left = np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=object)
    right = np.array([[1, 2, 3], [4, 5]], dtype=object)
    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)
    left = np.array([np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=object), np.array([np.array([6]), np.array([7, 8]), np.array([9])], dtype=object)], dtype=object)
    right = np.array([[[1, 2, 3], [4, 5]], [[6], [7, 8], [9]]], dtype=object)
    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)
    subarr = np.empty(2, dtype=object)
    subarr[:] = [np.array([None, 'b'], dtype=object), np.array(['c', 'd'], dtype=object)]
    left = np.array([subarr, None], dtype=object)
    right = np.array([[[None, 'b'], ['c', 'd']], None], dtype=object)
    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)