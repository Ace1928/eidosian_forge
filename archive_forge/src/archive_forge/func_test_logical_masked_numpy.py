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
@pytest.mark.parametrize('op, exp', [['__and__', True], ['__or__', True], ['__xor__', False]])
def test_logical_masked_numpy(self, op, exp):
    data = [True, False, None]
    ser_masked = pd.Series(data, dtype='boolean')
    ser_pa = pd.Series(data, dtype='boolean[pyarrow]')
    result = getattr(ser_pa, op)(ser_masked)
    expected = pd.Series([exp, False, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)