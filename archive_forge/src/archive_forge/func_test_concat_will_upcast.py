from collections import (
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tests.extension.decimal import to_decimal
@pytest.mark.parametrize('pdt', [Series, DataFrame])
def test_concat_will_upcast(pdt, any_signed_int_numpy_dtype):
    dt = any_signed_int_numpy_dtype
    dims = pdt().ndim
    dfs = [pdt(np.array([1], dtype=dt, ndmin=dims)), pdt(np.array([np.nan], ndmin=dims)), pdt(np.array([5], dtype=dt, ndmin=dims))]
    x = concat(dfs)
    assert x.values.dtype == 'float64'