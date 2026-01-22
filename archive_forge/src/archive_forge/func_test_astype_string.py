import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock
@pytest.mark.parametrize('nullable_string_dtype', ['string[python]', pytest.param('string[pyarrow]', marks=td.skip_if_no('pyarrow'))])
def test_astype_string(self, data, nullable_string_dtype):
    result = pd.Series(data[:5]).astype(nullable_string_dtype)
    expected = pd.Series([str(x) if not isinstance(x, bytes) else x.decode() for x in data[:5]], dtype=nullable_string_dtype)
    tm.assert_series_equal(result, expected)