import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['int64', 'uint64', 'int32', 'int16', 'int8', 'uint32', 'uint16', 'uint8'])
def test_interpolate_arrow(self, dtype):
    pytest.importorskip('pyarrow')
    df = DataFrame({'a': [1, None, None, None, 3]}, dtype=dtype + '[pyarrow]')
    result = df.interpolate(limit=2)
    expected = DataFrame({'a': [1, 1.5, 2.0, None, 3]}, dtype='float64[pyarrow]')
    tm.assert_frame_equal(result, expected)