from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_read_csv_with_use_inf_as_na(all_parsers):
    parser = all_parsers
    data = '1.0\nNaN\n3.0'
    msg = 'use_inf_as_na option is deprecated'
    warn = FutureWarning
    if parser.engine == 'pyarrow':
        warn = (FutureWarning, DeprecationWarning)
    with tm.assert_produces_warning(warn, match=msg, check_stacklevel=False):
        with option_context('use_inf_as_na', True):
            result = parser.read_csv(StringIO(data), header=None)
    expected = DataFrame([1.0, np.nan, 3.0])
    tm.assert_frame_equal(result, expected)