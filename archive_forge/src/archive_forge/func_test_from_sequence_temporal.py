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
@pytest.mark.parametrize('pa_type', tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES)
def test_from_sequence_temporal(pa_type):
    val = 3
    unit = pa_type.unit
    if pa.types.is_duration(pa_type):
        seq = [pd.Timedelta(val, unit=unit).as_unit(unit)]
    else:
        seq = [pd.Timestamp(val, unit=unit, tz=pa_type.tz).as_unit(unit)]
    result = ArrowExtensionArray._from_sequence(seq, dtype=pa_type)
    expected = ArrowExtensionArray(pa.array([val], type=pa_type))
    tm.assert_extension_array_equal(result, expected)