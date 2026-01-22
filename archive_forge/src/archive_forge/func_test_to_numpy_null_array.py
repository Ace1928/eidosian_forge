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
@pytest.mark.parametrize('na_val, exp', [(lib.no_default, np.nan), (1, 1)])
def test_to_numpy_null_array(na_val, exp):
    arr = pd.array([pd.NA, pd.NA], dtype='null[pyarrow]')
    result = arr.to_numpy(dtype='float64', na_value=na_val)
    expected = np.array([exp] * 2, dtype='float64')
    tm.assert_numpy_array_equal(result, expected)