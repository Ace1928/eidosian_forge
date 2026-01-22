import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_reindex_empty_series_tz_dtype():
    result = Series(dtype='datetime64[ns, UTC]').reindex([0, 1])
    expected = Series([NaT] * 2, dtype='datetime64[ns, UTC]')
    tm.assert_equal(result, expected)