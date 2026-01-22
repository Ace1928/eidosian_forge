from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@skip_pyarrow
@pytest.mark.parametrize('val', [np.iinfo(np.uint64).max + 1, np.iinfo(np.int64).min - 1])
def test_outside_int64_uint64_range(all_parsers, val):
    parser = all_parsers
    result = parser.read_csv(StringIO(str(val)), header=None)
    expected = DataFrame([str(val)])
    tm.assert_frame_equal(result, expected)