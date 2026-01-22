from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_reindex_uint_dtypes_fill_value(self, any_unsigned_int_numpy_dtype):
    df = DataFrame({'a': [1, 2], 'b': [1, 2]}, dtype=any_unsigned_int_numpy_dtype)
    result = df.reindex(columns=list('abcd'), index=[0, 1, 2, 3], fill_value=10)
    expected = DataFrame({'a': [1, 2, 10, 10], 'b': [1, 2, 10, 10], 'c': 10, 'd': 10}, dtype=any_unsigned_int_numpy_dtype)
    tm.assert_frame_equal(result, expected)