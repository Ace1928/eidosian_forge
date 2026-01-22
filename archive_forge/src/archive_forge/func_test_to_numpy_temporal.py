from __future__ import annotations
from datetime import (
from decimal import Decimal
from io import (
import operator
import pickle
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import timezones
from pandas.compat import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import (
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import no_default
from pandas.api.types import (
from pandas.tests.extension import base
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.arrow.extension_types import ArrowPeriodType
@pytest.mark.parametrize('pa_type', tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES, ids=repr)
@pytest.mark.parametrize('dtype', [None, object])
def test_to_numpy_temporal(pa_type, dtype):
    arr = ArrowExtensionArray(pa.array([1, None], type=pa_type))
    result = arr.to_numpy(dtype=dtype)
    if pa.types.is_duration(pa_type):
        value = pd.Timedelta(1, unit=pa_type.unit).as_unit(pa_type.unit)
    else:
        value = pd.Timestamp(1, unit=pa_type.unit, tz=pa_type.tz).as_unit(pa_type.unit)
    if dtype == object or (pa.types.is_timestamp(pa_type) and pa_type.tz is not None):
        if dtype == object:
            na = pd.NA
        else:
            na = pd.NaT
        expected = np.array([value, na], dtype=object)
        assert result[0].unit == value.unit
    else:
        na = pa_type.to_pandas_dtype().type('nat', pa_type.unit)
        value = value.to_numpy()
        expected = np.array([value, na])
        assert np.datetime_data(result[0])[0] == pa_type.unit
    tm.assert_numpy_array_equal(result, expected)