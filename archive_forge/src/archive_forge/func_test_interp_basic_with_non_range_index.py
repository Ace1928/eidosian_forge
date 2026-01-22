import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason="interpolate doesn't work for string")
def test_interp_basic_with_non_range_index(self, using_infer_string):
    df = DataFrame({'A': [1, 2, np.nan, 4], 'B': [1, 4, 9, np.nan], 'C': [1, 2, 3, 5], 'D': list('abcd')})
    msg = 'DataFrame.interpolate with object dtype'
    warning = FutureWarning if not using_infer_string else None
    with tm.assert_produces_warning(warning, match=msg):
        result = df.set_index('C').interpolate()
    expected = df.set_index('C')
    expected.loc[3, 'A'] = 3
    expected.loc[5, 'B'] = 9
    tm.assert_frame_equal(result, expected)