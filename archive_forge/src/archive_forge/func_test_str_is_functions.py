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
@pytest.mark.parametrize('value, method, exp', [['a1c', 'isalnum', True], ['!|,', 'isalnum', False], ['aaa', 'isalpha', True], ['!!!', 'isalpha', False], ['٠', 'isdecimal', True], ['~!', 'isdecimal', False], ['2', 'isdigit', True], ['~', 'isdigit', False], ['aaa', 'islower', True], ['aaA', 'islower', False], ['123', 'isnumeric', True], ['11I', 'isnumeric', False], [' ', 'isspace', True], ['', 'isspace', False], ['The That', 'istitle', True], ['the That', 'istitle', False], ['AAA', 'isupper', True], ['AAc', 'isupper', False]])
def test_str_is_functions(value, method, exp):
    ser = pd.Series([value, None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, method)()
    expected = pd.Series([exp, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)