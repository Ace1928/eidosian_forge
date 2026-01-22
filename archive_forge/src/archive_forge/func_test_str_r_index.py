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
@pytest.mark.parametrize('method', ['index', 'rindex'])
@pytest.mark.parametrize('start, end', [[0, None], [1, 4]])
def test_str_r_index(method, start, end):
    ser = pd.Series(['abcba', None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, method)('c', start, end)
    expected = pd.Series([2, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)
    with pytest.raises(ValueError, match='substring not found'):
        getattr(ser.str, method)('foo', start, end)